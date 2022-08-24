import os
import subprocess
import json
import utils
from typing import Any, Dict, List, Set, Tuple, IO

from setting import (
    JSON_FOLDER_BASE_DIR,
    LOG_DIR,
    MERGED_FOLDER_BASE_DIR,
    PROFILE_DIR,
    SUMMARY_FOLDER_DIR,
    Test,
    CompilerType,
    CoverageRecord,
    TestList,
    TestStatusType
)

from clang_coverage import LlvmCoverageParser
#------Parse the coverage result------

# coverage_records: Dict[str, LineInfo] = dict()
CoverageItem = Tuple[str, float, int, int]
covered_lines: Dict[str, Set[int]] = {}
uncovered_lines: Dict[str, Set[int]] = {}
tests_type = {"success": set(), "partial": set(), "fail": set()}

def get_json_obj(json_file: str):
    """
    Sometimes at the start of file llvm/gcov will complains "fail to find coverage data",
    then we need to skip these lines
      -- success read: 0      -  this json file have the full json coverage information
      -- partial success: 1   -  this json file starts with some error prompt, but still have the coverage information
      -- fail to read: 2      -  this json file doesn't have any coverage information
    """
    read_status = -1
    with open(json_file) as f:
        lines = f.readlines()
        for line in lines:
            try:
                json_obj = json.loads(line)
            except json.JSONDecodeError:
                read_status = 1
                continue
            else:
                if read_status == -1:
                    # not meet jsonDecoderError before, return success
                    read_status = 0
                return (json_obj, read_status)
    return None, 2

def update_coverage(
    coverage_records: List[CoverageRecord],
    interested_folders: List[str],
):
    for item in coverage_records:
        # extract information for the record
        record = item.to_dict()
        file_path = record["filepath"]
        if not utils.is_intrested_file(file_path, interested_folders):
            continue
        covered_range = record["covered_lines"]
        uncovered_range = record["uncovered_lines"]
        # transform file name: remote/13223/caffe2/aten -> caffe2/aten
        file_path = utils.transform_file_name(file_path, interested_folders)

        # if file not exists, add it into dictionary
        if file_path not in covered_lines:
            covered_lines[file_path] = set()
        if file_path not in uncovered_lines:
            uncovered_lines[file_path] = set()
        # update this file's covered and uncovered lines
        if covered_range is not None:
            covered_lines[file_path].update(covered_range)
        if uncovered_range is not None:
            uncovered_lines[file_path].update(uncovered_range)
            
def parse_json(json_file: str):
    print("start parse:", json_file)
    json_obj, read_status = get_json_obj(json_file)
    if read_status == 0:
        tests_type["success"].add(json_file)
    elif read_status == 1:
        tests_type["partial"].add(json_file)
    else:
        tests_type["fail"].add(json_file)
        raise RuntimeError(
            "Fail to do code coverage! Fail to load json file: ", json_file
        )

    cov_type = utils.detect_compiler_type()

    coverage_records: List[CoverageRecord] = []
    if cov_type == CompilerType.CLANG:
        coverage_records = LlvmCoverageParser(json_obj).parse("fbcode")
        #print(coverage_records)
    elif cov_type == CompilerType.GCC:
        coverage_records = None
        pass # FIXME
        #coverage_records = GcovCoverageParser(json_obj).parse()

    return coverage_records

def parse_jsons(
    test_list, interested_folders: List[str]
) -> None:
    g = os.walk(JSON_FOLDER_BASE_DIR)

    for path, _, file_list in g:
        for file_name in file_list:
            if file_name.endswith(".json"):
                # if compiler is clang, we only analyze related json / when compiler is gcc, we analyze all jsons
                cov_type = utils.detect_compiler_type()
                if cov_type == CompilerType.CLANG and not utils.related_to_test_list(
                    file_name, test_list
                ):
                    continue
                json_file = os.path.join(path, file_name)
                try:
                    coverage_records = parse_json(json_file)
                except RuntimeError:
                    print("Fail to load json file: ", json_file)
                    continue
                # collect information from each target's export file and merge them together:
                update_coverage(coverage_records, interested_folders)

def update_set() -> None:
    for file_name in covered_lines:
        # difference_update
        uncovered_lines[file_name].difference_update(covered_lines[file_name])

#------Print functionality------

def is_this_type_of_tests(target_name: str, test_set_by_type: Set[str]) -> bool:
    # tests are divided into three types: success / partial success / fail to collect coverage
    for test in test_set_by_type:
        if target_name in test:
            return True
    return False

def print_test_by_type(
    tests: TestList, test_set_by_type: Set[str], type_name: str, summary_file: IO[str]
) -> None:

    print("Tests " + type_name + " to collect coverage:", file=summary_file)
    for test in tests:
        if is_this_type_of_tests(test.name, test_set_by_type):
            print(test.name, file=summary_file)
    print(file=summary_file)


def print_test_condition(
    tests: TestList,
    tests_type: TestStatusType,
    interested_folders: List[str],
    coverage_only: List[str],
    summary_file: IO[str],
    summary_type: str,
) -> None:
    print_test_by_type(tests, tests_type["success"], "fully success", summary_file)
    print_test_by_type(tests, tests_type["partial"], "partially success", summary_file)
    print_test_by_type(tests, tests_type["fail"], "failed", summary_file)
    print(
        "\n\nCoverage Collected Over Interested Folders:\n",
        interested_folders,
        file=summary_file,
    )
    print(
        "\n\nCoverage Compilation Flags Only Apply To: \n",
        coverage_only,
        file=summary_file,
    )
    print(
        "\n\n---------------------------------- "
        + summary_type
        + " ----------------------------------",
        file=summary_file,
    )
    
def line_oriented_report(
    tests: TestList,
    tests_type: TestStatusType,
    interested_folders: List[str],
    coverage_only: List[str],
    covered_lines: Dict[str, Set[int]],
    uncovered_lines: Dict[str, Set[int]],
) -> None:
    with open(os.path.join(SUMMARY_FOLDER_DIR, "line_summary"), "w+") as report_file:
        print_test_condition(
            tests,
            tests_type,
            interested_folders,
            coverage_only,
            report_file,
            "LINE SUMMARY",
        )
        for file_name in covered_lines:
            covered = covered_lines[file_name]
            uncovered = uncovered_lines[file_name]
            print(
                f"{file_name}\n  covered lines: {sorted(covered)}\n  unconvered lines:{sorted(uncovered)}",
                file=report_file,
            )

def key_by_percentage(x: CoverageItem):
    return x[1]


def key_by_name(x: CoverageItem):
    return x[0]

def print_file_summary(
    covered_summary: int, total_summary: int, summary_file: IO[str]
) -> float:
    # print summary first
    try:
        coverage_percentage = 100.0 * covered_summary / total_summary
    except ZeroDivisionError:
        coverage_percentage = 0
    print(
        f"SUMMARY\ncovered: {covered_summary}\nuncovered: {total_summary}\npercentage: {coverage_percentage:.2f}%\n\n",
        file=summary_file,
    )
    if coverage_percentage == 0:
        print("Coverage is 0, Please check if json profiles are valid")
    return coverage_percentage

def print_file_oriented_report(
    tests_type: TestStatusType,
    coverage: List[CoverageItem],
    covered_summary: int,
    total_summary: int,
    summary_file: IO[str],
    tests: TestList,
    interested_folders: List[str],
    coverage_only: List[str],
) -> None:
    coverage_percentage = print_file_summary(
        covered_summary, total_summary, summary_file
    )
    # print test condition (interested folder / tests that are successsful or failed)
    print_test_condition(
        tests,
        tests_type,
        interested_folders,
        coverage_only,
        summary_file,
        "FILE SUMMARY",
    )
    # print each file's information
    for item in coverage:
        print(
            item[0].ljust(75),
            (str(item[1]) + "%").rjust(10),
            str(item[2]).rjust(10),
            str(item[3]).rjust(10),
            file=summary_file,
        )

    print(f"summary percentage:{coverage_percentage:.2f}%")
    
def file_oriented_report(
    tests: TestList,
    tests_type: TestStatusType,
    interested_folders: List[str],
    coverage_only: List[str],
    covered_lines: Dict[str, Set[int]],
    uncovered_lines: Dict[str, Set[int]],
) -> None:
    with open(os.path.join(SUMMARY_FOLDER_DIR, "file_summary"), "w+") as summary_file:
        covered_summary = 0
        total_summary = 0
        coverage = []
        for file_name in covered_lines:
            # get coverage number for this file
            covered_count = len(covered_lines[file_name])
            total_count = covered_count + len(uncovered_lines[file_name])
            try:
                percentage = round(covered_count / total_count * 100, 2)
            except ZeroDivisionError:
                percentage = 0
            # store information in a list to be sorted
            coverage.append((file_name, percentage, covered_count, total_count))
            # update summary
            covered_summary = covered_summary + covered_count
            total_summary = total_summary + total_count
        # sort
        coverage.sort(key=key_by_name)
        coverage.sort(key=key_by_percentage)
        # print
        print_file_oriented_report(
            tests_type,
            coverage,
            covered_summary,
            total_summary,
            summary_file,
            tests,
            interested_folders,
            coverage_only,
        )
             
def summarize_jsons(
    test_list,
    interested_folders: List[str],
    coverage_only: List[str],
):
    parse_jsons(test_list, interested_folders)
    update_set()
    line_oriented_report(
        test_list,
        tests_type,
        interested_folders,
        coverage_only,
        covered_lines,
        uncovered_lines,
    )
    file_oriented_report(
        test_list,
        tests_type,
        interested_folders,
        coverage_only,
        covered_lines,
        uncovered_lines,
    )
