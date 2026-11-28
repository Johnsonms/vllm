import torch, sys, itertools
from quack.epilogue.library import qk_rope_epi
from flashinfer.testing import bench_gpu_time_with_cupti
torch.manual_seed(0)
K, NQ, NKV, hd = [int(v) for v in sys.argv[1].split(',')]   # e.g. 4096,4096,1024,128
Ms=[int(v) for v in sys.argv[2].split(',')]
N=NQ+2*NKV; NQK=NQ+NKV; eps=1e-6
W=torch.randn(N,K,device='cuda',dtype=torch.bfloat16)/K**0.5
wq=torch.rand(hd,device='cuda',dtype=torch.bfloat16)+0.5; wk=torch.rand(hd,device='cuda',dtype=torch.bfloat16)+0.5
maxpos=8192
inv=1.0/(1e6**(torch.arange(0,hd,2,device='cuda').float()/hd))
fr=torch.outer(torch.arange(maxpos,device='cuda').float(),inv); cos_c=fr.cos(); sin_c=fr.sin()   # (P, hd/2)
def ref_fn(y, pos):   # vLLM semantics: neox rope, per-head weighted rmsnorm on q,k
    q,k,v=y.split([NQ,NKV,NKV],-1)
    def nr(t,w):
        t=t.view(t.shape[0],-1,hd).float(); t=t*torch.rsqrt(t.pow(2).mean(-1,keepdim=True)+eps)*w.float()
        c=cos_c[pos][:,None,:]; s=sin_c[pos][:,None,:]
        x1,x2=t[...,:hd//2],t[...,hd//2:]
        return torch.cat([x1*c-x2*s,x2*c+x1*s],-1).to(torch.bfloat16).flatten(1)
    return nr(q,wq),nr(k,wk),v
cref=torch.compile(ref_fn, dynamic=False)
# QuACK: interleaved rope -> permute each q/k head's rows so (i, i+hd/2) become adjacent (2i, 2i+1)
perm=torch.stack([torch.arange(hd//2),torch.arange(hd//2)+hd//2],1).flatten().cuda()
def permute_heads(Wpart):
    return Wpart.view(-1,hd,K)[:,perm,:].reshape(-1,K)
Wqk=torch.cat([permute_heads(W[:NQ]),permute_heads(W[NQ:NQK])]).contiguous()
Wv=W[NQK:].contiguous()
wvec=torch.cat([wq[perm].repeat(NQ//hd),wk[perm].repeat(NKV//hd)]).contiguous()
for M in Ms:
    x=torch.randn(M,K,device='cuda',dtype=torch.bfloat16)
    pos=torch.randint(0,maxpos,(M,),device='cuda')
    cs=torch.stack([cos_c[pos],sin_c[pos]],-1).flatten(1).to(torch.bfloat16).contiguous()  # (M,hd) interleaved
    # baseline: cuBLAS qkv + compiled norm+rope
    def base(x,pos): return cref(torch.nn.functional.linear(x,W),pos)
    q0,k0,v0=base(x,pos)
    Dqk=torch.empty(M,NQK,device='cuda',dtype=torch.bfloat16)
    tuned=qk_rope_epi.gemm_tuned(x,Wqk,Dqk,epi_args={"cs":cs,"qk":hd,"w":wvec})
    def quack(x,cs):
        qk_rope_epi.gemm(x,Wqk,Dqk,epi_args={"cs":cs,"qk":hd,"w":wvec},**cfgkw)
        return torch.nn.functional.linear(x,Wv)
    cfg=tuned.config if hasattr(tuned,'config') else tuned[1]
    cfgkw=dict(tile_M=cfg.tile_m,tile_N=cfg.tile_n,cluster_M=cfg.cluster_m,cluster_N=cfg.cluster_n,pingpong=cfg.pingpong,is_dynamic_persistent=cfg.is_dynamic_persistent)
    v1=quack(x,cs)
    Wall=torch.cat([Wqk,Wv]).contiguous(); Dall=torch.empty(M,N,device='cuda',dtype=torch.bfloat16); wall=torch.cat([wvec,torch.ones(NKV,device='cuda',dtype=torch.bfloat16)])
    tunedA=qk_rope_epi.gemm_tuned(x,Wall,Dall,epi_args={"cs":cs,"qk":hd,"w":wall}); cA=tunedA.config if hasattr(tunedA,'config') else tunedA[1]
    kwA=dict(tile_M=cA.tile_m,tile_N=cA.tile_n,cluster_M=cA.cluster_m,cluster_N=cA.cluster_n,pingpong=cA.pingpong,is_dynamic_persistent=cA.is_dynamic_persistent)
    def quack_all(x,cs): qk_rope_epi.gemm(x,Wall,Dall,epi_args={"cs":cs,"qk":hd,"w":wall},**kwA)
    def cublas_only(x): return torch.nn.functional.linear(x,W)
    # un-permute quack output for comparison
    qp=Dqk[:,:NQ].view(M,-1,hd); kp=Dqk[:,NQ:].view(M,-1,hd)
    inv_perm=torch.argsort(perm)
    q1=qp[...,inv_perm].flatten(1); k1=kp[...,inv_perm].flatten(1)
    # fp32 reference
    yf=(x.float()@W.float().t())
    def nrf(t,w):
        t=t.view(M,-1,hd); t=t*torch.rsqrt(t.pow(2).mean(-1,keepdim=True)+eps)*w.float()
        c=cos_c[pos][:,None,:]; s=sin_c[pos][:,None,:]; x1,x2=t[...,:hd//2],t[...,hd//2:]
        return torch.cat([x1*c-x2*s,x2*c+x1*s],-1).flatten(1)
    qf=nrf(yf[:,:NQ],wq); kf=nrf(yf[:,NQ:NQK],wk)
    rel=lambda a,b:((a.float()-b).norm()/b.norm()).item()
    tb=bench_gpu_time_with_cupti(base,input_args=(x,pos),use_cuda_graph=True,cold_l2_cache=True)
    tq=bench_gpu_time_with_cupti(quack,input_args=(x,cs),use_cuda_graph=True,cold_l2_cache=True)
    tA=bench_gpu_time_with_cupti(quack_all,input_args=(x,cs),use_cuda_graph=True,cold_l2_cache=True)
    tC=bench_gpu_time_with_cupti(cublas_only,input_args=(x,),use_cuda_graph=True,cold_l2_cache=True)
    import statistics as st
    mb,mq=st.median(tb)*1e3,st.median(tq)*1e3
    print(f"M={M:5d} base {mb:8.2f}us quack {mq:8.2f}us speedup {mb/mq:5.2f}x | 1-GEMM(V wrong) {st.median(tA)*1e3:7.2f}us {mb/(st.median(tA)*1e3):4.2f}x | cuBLAS qkv only {st.median(tC)*1e3:7.2f}us  cfg {cfg.tile_m}x{cfg.tile_n} c{cfg.cluster_m}x{cfg.cluster_n} pp{int(cfg.pingpong)} dyn{int(cfg.is_dynamic_persistent)}  err q base {rel(q0,qf):.4f} quack {rel(q1,qf):.4f}  k base {rel(k0,kf):.4f} quack {rel(k1,kf):.4f} v_eq {torch.equal(v0,v1)}", flush=True)
