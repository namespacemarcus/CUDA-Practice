import os


def count_lines(directory, extensions):
    total_lines = 0
    if not os.path.isdir(directory):
        return total_lines
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        non_empty = [line for line in lines if line.strip() != ""]
                        total_lines += len(non_empty)
                except Exception:
                    continue
    return total_lines


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))

    src_path = os.path.join(base_path, "../src")
    cu_lines = count_lines(src_path, [".cu", ".cuh"])

    bench_path = os.path.join(base_path, "../benchmarks")
    test_path = os.path.join(base_path, "../tests")
    bench_py = count_lines(bench_path, [".py"])
    test_py = count_lines(test_path, [".py"])
    total_py = bench_py + test_py

    total_all = cu_lines + total_py

    print("CUDA files (.cu, .cuh) in src:", cu_lines)
    print("Python files in benchmarks:", bench_py)
    print("Python files in tests:", test_py)
    print("Total Python lines:", total_py)
    print("Total lines of code:", total_all)
