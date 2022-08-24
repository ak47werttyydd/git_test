import os
import subprocess
import json
from typing import Any, Dict, List, Set, Tuple

from setting import (
    JSON_FOLDER_BASE_DIR,
    LOG_DIR,
    MERGED_FOLDER_BASE_DIR,
    PROFILE_DIR,
    SUMMARY_FOLDER_DIR,
    Test,
    CompilerType,
    CoverageRecord
)


#------init------i
def create_folder(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

def get_raw_profiles_folder() -> str:
    return os.environ.get("RAW_PROFILES_FOLDER", os.path.join(PROFILE_DIR, "raw"))

def create_folders():
    create_folder(
        PROFILE_DIR,
        MERGED_FOLDER_BASE_DIR,
        JSON_FOLDER_BASE_DIR,
        get_raw_profiles_folder(),
        SUMMARY_FOLDER_DIR,
        LOG_DIR,
    )

def convert_to_relative_path(whole_path: str, base_path: str) -> str:
    # ("profile/raw", "profile") -> "raw"
    if base_path not in whole_path:
        raise RuntimeError(base_path + " is not in " + whole_path)
    return whole_path[len(base_path) + 1 :]

def create_corresponding_folder(
    cur_path: str, prefix_cur_path: str, dir_list: List[str], new_base_folder: str
):
    for dir_name in dir_list:
        relative_path = convert_to_relative_path(
            cur_path, prefix_cur_path
        )  # get folder name like 'aten'
        new_folder_path = os.path.join(new_base_folder, relative_path, dir_name)
        create_folder(new_folder_path)

def detect_compiler_type():
    # check if user specifies the compiler type
    user_specify = os.environ.get("CXX", None)
    if user_specify:
        if user_specify in ["clang", "clang++"]:
            return CompilerType.CLANG
        elif user_specify in ["gcc", "g++"]:
            return CompilerType.GCC

        raise RuntimeError(f"User specified compiler is not valid {user_specify}")

    # auto detect
    auto_detect_result = subprocess.check_output(
        ["cc", "-v"], stderr=subprocess.STDOUT
    ).decode("utf-8")
    if "clang" in auto_detect_result:
        return CompilerType.CLANG
    elif "gcc" in auto_detect_result:
        return CompilerType.GCC
    raise RuntimeError(f"Auto detected compiler is not valid {auto_detect_result}")

#------Run the test file------
def get_llvm_tool_path() -> str:
    return os.environ.get(
        "LLVM_TOOL_PATH", "/usr/local/opt/llvm/bin"
    )  # set default as llvm path in dev server, on mac the default may be /usr/local/opt/llvm/bin


# a file is related if it's in one of the test_list folder
def related_to_test_list(file_name: str, test_list) -> bool:
    for test in test_list:
        if test.name in file_name:
            return True
    return False

def replace_extension(filename: str, ext: str) -> str:
    return filename[: filename.rfind(".")] + ext

def get_TVM_folder() -> str:
    return os.path.abspath(
        os.environ.get(
            "TVM_HOME"
        )
    )

def get_binary_file(test: str):
    binary_file = "python3 " + os.path.join(test.prefix, test.name)
    return binary_file

def get_shared_library():
    lib_dir = os.path.join(get_TVM_folder(), "build")
    result = [
        os.path.join(lib_dir, lib)
        for lib in os.listdir(lib_dir)
        if lib.endswith(".dylib")
    ]
    if len(result) == 0:
        result = [
        os.path.join(lib_dir, lib)
        for lib in os.listdir(lib_dir)
        if lib.endswith(".so")
        ]
    if len(result) == 0:
        raise Exception("No shared linrary found")
    return result

def run_python_test(binary_file: str):
    # python test script
    try:
        subprocess.check_call(
            binary_file, shell=True, cwd="./"
        )
    except subprocess.CalledProcessError:
        print(f"Binary failed to run: {binary_file}")

#------Function used in parse------
def is_intrested_file(
    file_path: str, interested_folders: List[str]
):
    #ignored_patterns = ["cuda", "aten/gen_aten", "aten/aten_", "build/"]
    ignored_patterns = ["NoIgnored"] # FIXME
    if any([pattern in file_path for pattern in ignored_patterns]):
        return False

    # ignore files that are not belong to TVM
    if not file_path.startswith(get_TVM_folder()):
        return False
    # if user has specifiled interested folder
    if interested_folders:
        for folder in interested_folders:
            intersted_folder_path = folder if folder.endswith("/") else f"{folder}/"
            if intersted_folder_path in file_path:
                return True
        return False
    else:
        return True

def transform_file_name(
    file_path: str, interested_folders: List[str]
) -> str:
    #remove_patterns: Set[str] = {".DEFAULT.cpp", ".AVX.cpp", ".AVX2.cpp"}
    remove_patterns: Set[str] = {"NotIgnored"} # FIXME
    for pattern in remove_patterns:
        file_path = file_path.replace(pattern, "")
    # if user has specifiled interested folder
    if interested_folders:
        for folder in interested_folders:
            if folder in file_path:
                return file_path[file_path.find(folder) :]
    # remove TVM base folder path
    TVM_foler = get_TVM_folder()
    assert file_path.startswith(TVM_foler)
    file_path = file_path[len(TVM_foler) + 1 :]
    return file_path



