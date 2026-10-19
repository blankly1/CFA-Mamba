"""Register hashes only after the user has verified checkpoint provenance and class order."""
import argparse
from contracts import PACKAGE, DOMAINS, CLASSES, settings, sha, read_json, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings',default=str(PACKAGE/'settings.json'))
    p.add_argument('--dataset',choices=list(DOMAINS),required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--class-names-json',required=True,help='Verified classifier output names in row order; JSON array')
    p.add_argument('--provenance',required=True,help='Restored archive or source-only training protocol and checkpoint selection')
    p.add_argument('--confirm-reviewed',action='store_true')
    args=p.parse_args()
    if not args.confirm_reviewed:
        p.error('Explicit --confirm-reviewed required; hashes alone cannot verify semantic class order')
    if args.source not in DOMAINS[args.dataset]:
        p.error('Invalid source domain')
    s=settings(args.settings)
    names=read_json(args.class_names_json)
    if not isinstance(names,list) or len(names)!=CLASSES[args.dataset] or len(set(names))!=len(names):
        raise ValueError('Invalid class list')
    files={}
    for part in ('F','B','C'):
        path=s['source_root']/args.dataset/args.source/f'source_{part}.pt'
        files[path.name]=sha(path)
    registry=read_json(s['source_bundles'])
    key=f'{args.dataset}/{args.source}'
    if key in registry:
        raise ValueError('Existing registry entry; create a new protocol/settings file instead of overwriting')
    registry[key]=dict(reviewed=True,provenance=args.provenance,class_names=names,sha256=files)
    write_json(s['source_bundles'],registry)
    print(key)


if __name__=='__main__':
    main()
