#!/usr/bin/env python3
"""Summarize tensors in a trusted PyTorch state_dict; emits metadata, not raw arrays."""
import argparse, json
def main():
    p=argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--top-k",type=int,default=8)
    a=p.parse_args()
    import torch
    obj=torch.load(a.checkpoint,map_location="cpu",weights_only=True)
    if not isinstance(obj,dict): raise SystemExit("Expected plain state_dict")
    out=[]
    for name,t in obj.items():
        if not torch.is_tensor(t): continue
        x=t.detach().float()
        finite=torch.isfinite(x)
        y=x[finite]
        if y.numel():
            qs=torch.quantile(y,torch.tensor([0,.01,.1,.5,.9,.99,1.])).tolist()
            flat=y.abs().flatten()
            k=min(a.top_k,flat.numel())
            vals,idx=torch.topk(flat,k)
            top=[{"flat_index":int(i),"abs_value":float(v)} for v,i in zip(vals,idx)]
            rec={"key":name,"shape":list(t.shape),"dtype":str(t.dtype),"numel":t.numel(),
                 "finite_fraction":float(finite.float().mean()),"min":float(y.min()),
                 "max":float(y.max()),"mean":float(y.mean()),"std":float(y.std(unbiased=False)),
                 "l2_norm":float(torch.linalg.vector_norm(y)),"zero_fraction":float((y==0).float().mean()),
                 "quantiles_0_1_10_50_90_99_100":qs,"top_abs":top}
        else: rec={"key":name,"shape":list(t.shape),"dtype":str(t.dtype),"numel":t.numel(),"finite_fraction":0}
        out.append(rec)
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()
