import os


def batch_modify_include(root_dir: str):
    # 只扫描C/CUDA源码文件
    suffix = {".h", ".hpp", ".cuh", ".cu", ".cpp"}
    cnt_file = 0
    cnt_line = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            path = os.path.join(dirpath, fname)
            if not fname.endswith(tuple(suffix)):
                continue
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            modified = False
            new_lines = []
            for line in lines:
                old_line = line
                # 正则匹配 #include "../common/cuda/任意内容"
                if '#include "../common/cuda/' in line:
                    line = line.replace("../common/cuda/", "../common/")
                    cnt_line += 1
                    modified = True
                new_lines.append(line)
            if modified:
                with open(path, "w", encoding="utf-8", errors="ignore") as f:
                    f.writelines(new_lines)
                cnt_file += 1
                print(f"[修改] {path}")
    print(f"\n完成：修改文件 {cnt_file} 个，匹配替换行 {cnt_line} 条")


if __name__ == "__main__":
    # 修改此处路径为你的src文件夹绝对/相对路径
    SRC_PATH = "./src"
    batch_modify_include(SRC_PATH)
