"""Sample-trained full PCA with eigenvalues and verifiable projected-data provenance.

Input/output bin format: uint32 rows,cols followed by row-major float32.
Training saves one inseparable eigenvalue/rotation/mean model. No coordinates
are independently sorted after projection. Existing outputs are never replaced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def matrix(path):
    with open(path, 'rb') as f:
        n,d=struct.unpack('<II',f.read(8))
    if not n or not d or Path(path).stat().st_size != 8+n*d*4:
        raise ValueError('Invalid float32 bin shape/size')
    return np.memmap(path,dtype='<f4',mode='r',offset=8,shape=(n,d))


def load_model(path):
    with np.load(path,allow_pickle=False) as z:
        model={k:z[k].copy() for k in z.files}
    if int(model['version'])!=1:
        raise ValueError('Unsupported PCA model')
    v,rot,mean=model['eigenvalues'],model['rotation'],model['mean']
    d=len(v)
    if (v.shape!=(d,) or rot.shape!=(d,d) or mean.shape!=(d,) or
        not all(np.isfinite(a).all() for a in (v,rot,mean)) or
        np.any(v<0) or np.any(v[1:]>v[:-1]) or int(model['sample_count'])<2):
        raise ValueError('Invalid sample-trained PCA model')
    if not np.allclose(rot.T@rot,np.eye(d),rtol=0,atol=1e-8):
        raise ValueError('PCA rotation is not orthonormal')
    return model


def train(base,model_path,train_size=50000,seed=42):
    if train_size<2 or seed<0:
        raise ValueError('train-size must be >=2 and seed nonnegative')
    n,d=base.shape;k=min(n,train_size)
    if k<2:raise ValueError('PCA requires at least two training rows')
    ids=np.random.default_rng(seed).choice(n,k,replace=False)
    sample=np.array(base[ids],dtype=np.float64)
    if not np.isfinite(sample).all():raise ValueError('Non-finite training data')
    mean=sample.mean(axis=0);sample-=mean
    covariance=sample.T@sample/(k-1)  # Sample covariance eigenvalues, ddof=1.
    values,rotation=np.linalg.eigh(covariance)
    order=np.argsort(values)[::-1];values=values[order];rotation=rotation[:,order]
    if values.min() < -1e-10*max(1.0,float(values.max())):
        raise ValueError('Unexpected negative covariance eigenvalue')
    values=np.maximum(values,0)
    # Deterministic sign convention; eigenvalue and vector order stay paired.
    for j in range(d):
        if rotation[np.argmax(np.abs(rotation[:,j])),j]<0:rotation[:,j]*=-1
    with open(model_path,'xb') as f:
        np.savez(f,version=np.int64(1),eigenvalues=values,rotation=rotation,mean=mean,
                 sample_ids=ids,sample_count=np.int64(k),seed=np.int64(seed),
                 training_rows=np.int64(n),covariance_ddof=np.int64(1))


def transform(base,model_path,output,batch_size=8192):
    model=load_model(model_path);n,d=base.shape
    if len(model['mean'])!=d or batch_size<=0:raise ValueError('Invalid transform dimensions/batch size')
    output=Path(output);sidecar=Path(str(output)+'.pca.json')
    if output.exists() or sidecar.exists():raise FileExistsError('Projection output or provenance exists')
    sha=hashlib.sha256()
    with output.open('xb') as f:
        header=struct.pack('<II',n,d);f.write(header);sha.update(header)
        for lo in range(0,n,batch_size):
            x=np.asarray(base[lo:lo+batch_size],dtype=np.float64)
            if not np.isfinite(x).all():raise ValueError('Non-finite transform input')
            y=np.asarray((x-model['mean'])@model['rotation'],dtype='<f4')
            if not np.isfinite(y).all():raise ValueError('Projection exceeds float32 range')
            blob=y.tobytes();f.write(blob);sha.update(blob)
    with sidecar.open('x') as f:
        json.dump(dict(version=1,rows=n,dimensions=d,pca_sha256=sha.hexdigest(),
                       model_sha256=digest(model_path),weight_source='sample_pca_eigenvalues',
                       sample_count=int(model['sample_count']),covariance_ddof=1),f,indent=2)


def allocation_eigenvalues(model_path,pca_path):
    """Reject eigenvalues from a model other than the one projecting this file."""
    model=load_model(model_path);base=matrix(pca_path)
    meta=json.loads(Path(str(pca_path)+'.pca.json').read_text())
    if (meta['version']!=1 or meta['model_sha256']!=digest(model_path) or
        meta['pca_sha256']!=digest(pca_path) or
        (meta['rows'],meta['dimensions'])!=base.shape or
        len(model['eigenvalues'])!=base.shape[1]):
        raise ValueError('PCA model/projection provenance mismatch')
    return model['eigenvalues'],meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    for mode in ('train','transform'):
        p=sub.add_parser(mode);p.add_argument('--base',required=True)
        p.add_argument('--pca-output',required=True);p.add_argument('--batch-size',type=int,default=8192)
        if mode=='train':
            p.add_argument('--model-output',required=True)
            p.add_argument('--train-size',type=int,default=50000);p.add_argument('--seed',type=int,default=42)
        else:p.add_argument('--model',required=True)
    args=parser.parse_args();base=matrix(args.base)
    try:
        if args.batch_size<=0:raise ValueError('batch-size must be positive')
        if Path(args.pca_output).exists() or Path(args.pca_output+'.pca.json').exists():
            raise FileExistsError('Projection output or provenance exists')
        if args.mode=='train':
            train(base,args.model_output,args.train_size,args.seed);model=args.model_output
        else:model=args.model
        transform(base,model,args.pca_output,args.batch_size)
    except (ValueError,OSError) as e:parser.error(str(e))
    print(json.dumps(dict(model=model,pca_base=args.pca_output,rows=len(base),dimensions=base.shape[1])))


if __name__=='__main__':main()
