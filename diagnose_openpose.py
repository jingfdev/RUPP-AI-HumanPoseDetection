import os
import sys


def main() -> int:
    print("Python:", sys.version)
    print("OPENPOSE_HOME=", os.environ.get("OPENPOSE_HOME"))
    print("OPENPOSE_PYTHON_PATH=", os.environ.get("OPENPOSE_PYTHON_PATH"))
    print("OPENPOSE_BIN_PATH=", os.environ.get("OPENPOSE_BIN_PATH"))

    try:
        import openpose_wrapper as ow
    except Exception as e:
        print("Failed to import openpose_wrapper:", repr(e))
        return 2

    print("openpose_wrapper imported OK")

    try:
        op = ow.import_pyopenpose()
        print("pyopenpose imported OK")
        print("pyopenpose module:", op)
        print("pyopenpose __file__:", getattr(op, "__file__", None))
        return 0
    except Exception as e:
        print("pyopenpose import FAILED:", repr(e))
        print("sys.path (first 20):")
        for p in sys.path[:20]:
            print("  ", p)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
