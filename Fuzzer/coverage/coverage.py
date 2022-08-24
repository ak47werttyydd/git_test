# <project folder>
import os
from typing import List, Any, Dict, Tuple, Optional
HOME_DIR = os.environ["HOME"]

TOOLS_FOLDER = os.path.dirname(os.path.realpath(__file__))
#TOOLS_FOLDER = os.path.join(
#    os.path.dirname(os.path.realpath(__file__)), os.path.pardir, os.path.pardir
#)


# <profile folder>
PROFILE_DIR = os.path.join(TOOLS_FOLDER, "profile")
JSON_FOLDER_BASE_DIR = os.path.join(PROFILE_DIR, "json")
MERGED_FOLDER_BASE_DIR = os.path.join(PROFILE_DIR, "merged")
SUMMARY_FOLDER_DIR = os.path.join(PROFILE_DIR, "summary")

# <log path>
LOG_DIR = os.path.join(PROFILE_DIR, "log")

def create_folder(*paths: Any):
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
    
def get_json_report(test_list: TestList, options: Option):
    cov_type = detect_compiler_type()
    check_compiler_type(cov_type)
    if cov_type == CompilerType.CLANG:
        # run
        if options.need_run:
            clang_run(test_list)
        # merge && export
        if options.need_merge:
            clang_coverage.merge(test_list, TestPlatform.OSS)
        if options.need_export:
            clang_coverage.export(test_list, TestPlatform.OSS)
    elif cov_type == CompilerType.GCC:
        # run
        if options.need_run:
            gcc_run(test_list)

# FIXME
def get_oss_binary_folder(test_type: TestType) -> str:
    assert test_type in {TestType.CPP, TestType.PY}
    # TODO: change the way we get binary file -- binary may not in build/bin ?
    return os.path.join(
        get_pytorch_folder(), "build/bin" if test_type == TestType.CPP else "test"
    )
    
def get_oss_binary_file(test_name: str, test_type: TestType) -> str:
    assert test_type in {TestType.CPP, TestType.PY}
    binary_folder = get_oss_binary_folder(test_type)
    binary_file = os.path.join(binary_folder, test_name)
    if test_type == TestType.PY:
        # add python to the command so we can directly run the script by using binary_file variable
        binary_file = "python " + binary_file
    return binary_file

def clang_run(tests: TestList):
    start_time = time.time()
    for test in tests:
        # raw_file
        raw_file = os.path.join(get_raw_profiles_folder(), test.name + ".profraw")
        # binary file
        binary_file = get_oss_binary_file(test.name, test.test_type)
        clang_coverage.run_target(
            binary_file, raw_file, test.test_type, TestPlatform.OSS
        )
    print_time("running binaries takes time: ", start_time, summary_time=True)


def gcc_run(tests: TestList):
    start_time = time.time()
    for test in tests:
        # binary file
        binary_file = get_oss_binary_file(test.name, test.test_type)
        gcc_coverage.run_target(binary_file, test.test_type)
    print_time("run binaries takes time: ", start_time, summary_time=True)


