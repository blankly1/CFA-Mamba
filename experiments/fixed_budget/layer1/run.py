import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'common'))
from launch import main

if __name__ == '__main__':
    sys.argv[1:1] = ['--layer', '1']
    main()
