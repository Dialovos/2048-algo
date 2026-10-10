// Same 5-bit board, n-tuple layouts, and backward TD(0) as the JS engine.
// One CUDA warp owns a game; each lane reads every 32nd feature. Kernels advance a bounded number of transitions;
// unfinished episodes stay on-device, so saving/stopping never waits a full game.
// LEVELS (set by train-gpu.py) picks the layout: 0 standard; 1 large (8 patterns
// in 3 stages); 2 extra large (8 patterns in 5 stages). Both large ones add the corner bonus.
typedef unsigned int U;
#ifndef LEVELS
#define LEVELS 0
#endif
const int PATTERNS = LEVELS ? 8 : 4, STAGES = 1 + 2 * LEVELS;
const int POOLS = LEVELS == 2 ? 6 : 5;
const int FEATURE_COUNT = PATTERNS * 8;
__device__ int FEATURES[FEATURE_COUNT * 6];
const int NW = (STAGES * PATTERNS) << 24;
const int POOL_SIZE = 1000;

struct Board { U r[4]; };
__device__ int rank_at(Board b, int c) { return (b.r[c>>2] >> (5*(c&3))) & 31; }
__device__ U column(Board b, int c) {
    return ((b.r[0] >> (5*c)) & 31) | (((b.r[1] >> (5*c)) & 31) << 5)
        | (((b.r[2] >> (5*c)) & 31) << 10) | (((b.r[3] >> (5*c)) & 31) << 15);
}
__device__ U reverse_row(U row) {
    return ((row & 31) << 15) | (((row >> 5) & 31) << 10)
        | (((row >> 10) & 31) << 5) | (row >> 15);
}
extern "C" __global__ void row_tables(U *left, U *right, U *rewards) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= (1 << 20)) return;
    int line[4] = {row & 31, (row >> 5) & 31, (row >> 10) & 31, row >> 15};
    U reward = 0;
    for (int i=0; i<3; i++) {
        int j=i+1; while (j<4 && line[j]==0) j++;
        if (j==4) break;
        if (line[i]==0) { line[i]=line[j]; line[j]=0; i--; }
        else if (line[i]==line[j]) {
            if (line[i]!=31) line[i]++;
            reward += 1u << line[i]; line[j]=0;
        }
    }
    U result=line[0] | (line[1]<<5) | (line[2]<<10) | (line[3]<<15);
    left[row]=result; right[reverse_row(row)]=reverse_row(result); rewards[row]=reward;
}
__device__ bool move_board(Board b, int dir, Board &out, U &reward,
                          const U *left, const U *right, const U *rewards) {
    reward=0;
    const U *table=(dir==0 || dir==3) ? left : right;
    if (dir==1 || dir==3) {
        #pragma unroll
        for (int i=0; i<4; i++) { out.r[i]=table[b.r[i]]; reward+=rewards[b.r[i]]; }
    } else {
        Board transposed;
        #pragma unroll
        for (int i=0; i<4; i++) { U c=column(b,i); transposed.r[i]=table[c]; reward+=rewards[c]; }
        #pragma unroll
        for (int i=0; i<4; i++) out.r[i]=column(transposed,i);
    }
    return out.r[0]!=b.r[0] || out.r[1]!=b.r[1] || out.r[2]!=b.r[2] || out.r[3]!=b.r[3];
}
// The board's stage, as in the JS engine: before 16384, then for each big-tile
// level (16384, then 32768) without and with another tile of 8192 or more.
__device__ int stage_of(Board b) {
#if LEVELS
    int top=0, big=0;
    for (int c=0; c<16; c++) { int r=min(rank_at(b,c),15); top=max(top,r); big+=r>=13; }
    return top<14 ? 0 : 1 + 2*(min(top,13+LEVELS)-14) + (big>=2);
#else
    return 0;
#endif
}
// Large layout: the corner bonus (potential in the JS engine) added to every value.
__device__ double potential(Board b) {
#if LEVELS
    const int corners[12]={0,1,4, 3,2,7, 12,13,8, 15,14,11}; // corner, then its edge neighbours
    int top=0;
    for (int c=0; c<16; c++) top=max(top,rank_at(b,c));
    double bonus=0;
    for (int k=0; k<12; k+=3) {
        if (rank_at(b,corners[k])!=top) continue;
        int next=max(rank_at(b,corners[k+1]),rank_at(b,corners[k+2]));
        bonus=fmax(bonus,ldexp(1.0,top)+(next ? ldexp(1.0,next)/2 : 0));
    }
    return bonus;
#else
    return 0;
#endif
}
__device__ int feature(Board b, int f, int stage) {
    int idx=(stage*PATTERNS+(f>>3))<<24;
    #pragma unroll
    for (int j=0; j<6; j++) {
        int c=FEATURES[f*6+j], rank=(b.r[c>>2] >> (5*(c&3))) & 31;
        idx |= min(rank,15) << (4*j);
    }
    return idx;
}
__device__ double value(Board b, const float *w) {
    int stage=stage_of(b); double sum=0;
    for (int f=0; f<FEATURE_COUNT; f++) sum+=(double)w[feature(b,f,stage)];
    return sum+potential(b);
}
// Atomic add returning the previous value. Unlike unsynchronized CPU workers,
// concurrent GPU updates are not lost.
#ifdef EXACT_ATOMICS
// Parity-test build: a CAS loop with a double intermediate matches JS
// Float32Array stores bit for bit, but in training most updates collide with
// another game's and retry, which made it the main cost of TC learning.
__device__ float add_weight(float *address, double increment) {
    int *bits=(int*)address;
    int old=atomicCAS(bits,0,0), assumed;
    do {
        assumed=old;
        float result=(float)((double)__int_as_float(assumed)+increment);
        old=atomicCAS(bits,assumed,__float_as_int(result));
    } while (old!=assumed);
    return __int_as_float(old);
}
#else
// Training build: hardware float atomics never retry. Rounding the increment to
// float first can change the last bit of a stored value relative to JS.
__device__ float add_weight(float *address, double increment) {
    return atomicAdd(address,(float)increment);
}
#endif
__device__ double learn(Board b, double target, double alpha, float *w, float *e, float *a, int tc) {
    int stage=stage_of(b), indexes[FEATURE_COUNT]; double sum=0, bonus=potential(b);
    for (int f=0; f<FEATURE_COUNT; f++) { indexes[f]=feature(b,f,stage); sum+=(double)w[indexes[f]]; }
    double error=target-(sum+bonus), step=alpha*error/FEATURE_COUNT;
    for (int f=0; f<FEATURE_COUNT; f++) {
        int i=indexes[f];
        if (tc) {
            double old_a=add_weight(a+i,fabs(error));
            double old_e=add_weight(e+i,error);
            add_weight(w+i,old_a==0 ? step : step*fabs(old_e)/old_a);
        } else add_weight(w+i,step);
    }
    sum=0;
    for (int f=0; f<FEATURE_COUNT; f++) sum+=(double)w[indexes[f]];
    return sum+bonus;
}
__device__ U random_bits(U &rng) { rng ^= rng<<13; rng ^= rng>>17; rng ^= rng<<5; return rng; }
__device__ double random_unit(U &rng) { return (double)random_bits(rng)/4294967296.0; }
__device__ void spawn(Board &b, U &rng) {
    int empty[16], count=0;
    for (int c=0; c<16; c++) if (((b.r[c>>2]>>(5*(c&3)))&31)==0) empty[count++]=c;
    if (!count) return;
    int c=empty[(int)(random_unit(rng)*count)];
    U rank=random_unit(rng)<0.9 ? 1 : 2;
    b.r[c>>2] |= rank << (5*(c&3));
}
__device__ int top_rank(Board b) {
    int top=0;
    for (int c=0; c<16; c++) top=max(top,(int)((b.r[c>>2]>>(5*(c&3)))&31));
    return top;
}
// Optimism is spread over the standard patterns (the first four of each
// stage); the large layout's extra patterns start at 0.
extern "C" __global__ void initialize_weights(float *w, float optimism) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if (i<NW) w[i]=((i>>24)%PATTERNS>=4 || (i & (i>>1) & (i>>2) & 0x222222)) ? 0 : optimism/32;
}

// Restart pools hold boards where a game first reached something. Standard:
// 8192, 16384, 32768, 65536, 131072 (pool = rank - 13). Large: 8192, 16384,
// stage 3, 32768, 65536. Extra large: 8192, 16384, stage 3, 32768, stage 5,
// 65536. The pick weights favour the late stages: 2x from 16384 on (large);
// 2x for the 16384 level and 4x for the 32768 level (extra large).
__device__ int pool_for_rank(int rank) {
#if LEVELS == 2
    return rank==13 ? 0 : rank==14 ? 1 : rank==15 ? 3 : rank==16 ? 5 : -1;
#elif LEVELS == 1
    return rank==13 ? 0 : rank==14 ? 1 : rank==15 ? 3 : rank==16 ? 4 : -1;
#else
    return rank>=13 && rank<=17 ? rank-13 : -1;
#endif
}
__device__ int pool_for_stage(int stage) {
    return LEVELS && (stage==2 || (LEVELS==2 && stage==4)) ? stage : -1;
}
__device__ int pool_weight(int k) {
#if LEVELS == 2
    const int weights[POOLS]={1,2,2,4,4,1};
    return weights[k];
#elif LEVELS == 1
    return k ? 2 : 1;
#else
    return 1;
#endif
}
__device__ void keep(Board b, int k, U *pools, U *counts, int n, int g, int lane, U &rng) {
    U count=counts[k*n+g];
    int p=count<POOL_SIZE ? count : (int)(random_unit(rng)*POOL_SIZE);
    if (lane==0) {
        for (int r=0;r<4;r++) pools[((k*POOL_SIZE+p)*4+r)*n+g]=b.r[r];
        counts[k*n+g]=min(count+1,(U)POOL_SIZE);
    }
}

// state is SoA: board[4], phase, forward length, reverse index, score, top,
// start pool code, PRNG, highest stage reached. phase 0=new game; 1=forward; 2=backward.
__device__ double warp_sum(double v) {
    for (int offset=16; offset; offset/=2) v+=__shfl_down_sync(0xffffffff,v,offset);
    return __shfl_sync(0xffffffff,v,0);
}
__device__ double warp_value(Board b, const float *w, int lane) {
    int stage=stage_of(b); double v=0;
    for (int f=lane; f<FEATURE_COUNT; f+=32) v+=(double)w[feature(b,f,stage)];
    return warp_sum(v)+potential(b);
}
__device__ double warp_learn(Board b, double target, double alpha, float *w, float *e, float *a, int tc, int lane) {
    const int PER_LANE=FEATURE_COUNT/32;
    int stage=stage_of(b), idx[PER_LANE]; double sum=0, bonus=potential(b);
    for (int k=0; k<PER_LANE; k++) { idx[k]=feature(b,lane+32*k,stage); sum+=(double)w[idx[k]]; }
    double error=target-(warp_sum(sum)+bonus), step=alpha*error/FEATURE_COUNT;
    for (int k=0; k<PER_LANE; k++) {
        int i=idx[k];
        unsigned peers=__match_any_sync(0xffffffff,i);
        // Symmetric patterns may address the same weight. Every update in one
        // step is identical, so lanes sharing an index let one lane apply them
        // in turn; independent indexes run in parallel.
        if (lane==__ffs(peers)-1) {
            for (int n=0; n<__popc(peers); n++) {
                if (tc) {
                    double old_a=add_weight(a+i,fabs(error));
                    double old_e=add_weight(e+i,error);
                    add_weight(w+i,old_a==0 ? step : step*fabs(old_e)/old_a);
                } else add_weight(w+i,step);
            }
        }
        __syncwarp();
    }
    sum=0;
    for (int k=0; k<PER_LANE; k++) sum+=(double)w[idx[k]];
    return warp_sum(sum)+bonus;
}

extern "C" __global__ void advance(
    float *w, float *e, float *a, int tc, const U *left, const U *right, const U *rewards,
    U *state, double *targets, U *history, U *pools, U *counts, U *results,
    unsigned long long *moves, int *overflow, int n, int capacity, int transitions,
    double alpha, double restart) {
    int lane=threadIdx.x & 31;
    int g=(blockDim.x*blockIdx.x+threadIdx.x)/32;
    if (g>=n) return;
    Board b; for (int r=0; r<4; r++) b.r[r]=state[r*n+g];
    int phase=state[4*n+g], length=state[5*n+g], back=(int)state[6*n+g];
    U score=state[7*n+g]; int top=state[8*n+g], start=state[9*n+g]; U rng=state[10*n+g];
    int stage_top=state[11*n+g];
    double target=targets[g]; unsigned long long progressed=0;
    if (lane==0) results[g]=0;
    for (int iteration=0; iteration<transitions; iteration++) {
        __syncwarp(); // make the owning lane's previous history/pool stores visible
        if (phase==0) {
            start=0; length=0; score=0;
            int available[POOLS], available_count=0, total=0;
            for (int k=0;k<POOLS;k++) if (counts[k*n+g]) { available[available_count++]=k; total+=pool_weight(k); }
            if (available_count && random_unit(rng)<restart) {
                int pick=(int)(random_unit(rng)*total), k=0;
                for (int j=0;j<available_count;j++) { k=available[j]; pick-=pool_weight(k); if (pick<0) break; }
                int p=(int)(random_unit(rng)*min(counts[k*n+g],(U)POOL_SIZE));
                for (int r=0;r<4;r++) b.r[r]=pools[((k*POOL_SIZE+p)*4+r)*n+g];
                start=k+13;
            } else {
                for (int r=0;r<4;r++) b.r[r]=0;
                spawn(b,rng); spawn(b,rng);
            }
            top=top_rank(b); stage_top=stage_of(b); phase=1;
        }
        if (phase==1) {
            if (length>=capacity) { if (lane==0) atomicMax(overflow,length+1); break; }
            Board best_board; U best_reward=0; bool valid=false; double best=-1e300;
            for (int dir=0; dir<4; dir++) {
                Board next; U reward;
                if (!move_board(b,dir,next,reward,left,right,rewards)) continue;
                double v=reward+warp_value(next,w,lane);
                if (v>best) { valid=true; best=v; best_reward=reward; best_board=next; }
            }
            if (!valid) { phase=2; back=length-1; target=0; }
            else {
                if (lane==0) {
                    for (int r=0;r<4;r++) history[(length*5+r)*n+g]=best_board.r[r];
                    history[(length*5+4)*n+g]=best_reward;
                }
                length++; progressed++; score+=best_reward;
                b=best_board; spawn(b,rng);
                int rank=top_rank(b);
                if (rank>top) {
                    top=rank;
                    int k=pool_for_rank(rank);
                    if (k>=0) keep(b,k,pools,counts,n,g,lane,rng);
                }
#if LEVELS
                int stage=stage_of(b);
                if (stage>stage_top) {
                    stage_top=stage;
                    int k=pool_for_stage(stage);
                    if (k>=0) keep(b,k,pools,counts,n,g,lane,rng);
                }
#endif
                continue;
            }
        }
        if (phase==2) {
            if (back>=0) {
                Board after; for (int r=0;r<4;r++) after.r[r]=history[(back*5+r)*n+g];
                target=history[(back*5+4)*n+g]+warp_learn(after,target,alpha,w,e,a,tc,lane);
                back--;
            }
            if (back<0) {
                if (lane==0) {
                    results[g]=1; results[n+g]=start; results[2*n+g]=score;
                    results[3*n+g]=1u<<top; results[4*n+g]=length;
                }
                phase=0; break; // host collects every completion, including zero-move games
            }
        }
    }
    if (lane!=0) return;
    for (int r=0;r<4;r++) state[r*n+g]=b.r[r];
    state[4*n+g]=phase; state[5*n+g]=length; state[6*n+g]=back;
    state[7*n+g]=score; state[8*n+g]=top; state[9*n+g]=start; state[10*n+g]=rng; state[11*n+g]=stage_top;
    targets[g]=target; moves[g]+=progressed;
}

// Small deterministic kernels used by the CPU/GPU parity checks.
extern "C" __global__ void inspect_boards(const U *boards, U *out, double *values,
                                         const float *w, const U *left, const U *right,
                                         const U *rewards, int n) {
    int g=blockIdx.x*blockDim.x+threadIdx.x; if(g>=n) return;
    Board b; for(int r=0;r<4;r++) b.r[r]=boards[g*4+r];
    values[g]=value(b,w);
    for(int dir=0;dir<4;dir++) {
        Board next; U reward; bool changed=move_board(b,dir,next,reward,left,right,rewards);
        for(int r=0;r<4;r++) out[(g*4+dir)*6+r]=next.r[r];
        out[(g*4+dir)*6+4]=reward; out[(g*4+dir)*6+5]=changed;
    }
}
extern "C" __global__ void learn_sequence(const U *boards, const double *targets, int n,
                                         double alpha, float *w, float *e, float *a, int tc) {
    if(blockIdx.x || threadIdx.x) return;
    for(int i=0;i<n;i++) {
        Board b; for(int r=0;r<4;r++) b.r[r]=boards[i*4+r];
        learn(b,targets[i],alpha,w,e,a,tc);
    }
}
