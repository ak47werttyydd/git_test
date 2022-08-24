import os
import subprocess
from typing import List, Optional, Tuple, Dict, Any, Set
import typing as t

import utils
from setting import (
    JSON_FOLDER_BASE_DIR,
    LOG_DIR,
    MERGED_FOLDER_BASE_DIR,
    PROFILE_DIR,
    SUMMARY_FOLDER_DIR,
    Test,
    CoverageRecord
)

def run_target(
    binary_file: str, raw_file: str
):
    print("start run: ", binary_file)
    # set environment variable -- raw profile output path of the binary run
    os.environ["LLVM_PROFILE_FILE"] = raw_file
    # run binary
    utils.run_python_test(binary_file)

        
def merge_target(raw_file: str, merged_file: str):
    print("start to merge target: ", raw_file)
    # run command
    llvm_tool_path = utils.get_llvm_tool_path()
    subprocess.check_call(
        [
            f"{llvm_tool_path}/llvm-profdata",
            "merge",
            "-sparse",
            raw_file,
            "-o",
            merged_file,
        ]
    )
    
def merge(test_list):
    print("start merge")
    # find all raw profile under raw_folder and sub-folders
    raw_folder_path = utils.get_raw_profiles_folder()
    g = os.walk(raw_folder_path)
    for path, dir_list, file_list in g:
        # if there is a folder raw/aten/, create corresponding merged folder profile/merged/aten/ if not exists yet
        utils.create_corresponding_folder(
            path, raw_folder_path, dir_list, MERGED_FOLDER_BASE_DIR
        )
        # check if we can find raw profile under this path's folder
        for file_name in file_list:
            if file_name.endswith(".profraw"):
                if not utils.related_to_test_list(file_name, test_list):
                    continue
                print(f"start merge {file_name}")
                raw_file = os.path.join(path, file_name)
                merged_file_name = utils.replace_extension(file_name, ".merged")
                merged_file = os.path.join(
                    MERGED_FOLDER_BASE_DIR,
                    utils.convert_to_relative_path(path, raw_folder_path),
                    merged_file_name,
                )
                merge_target(raw_file, merged_file)

def export_target(
    merged_file: str,
    json_file: str,
    binary_file: str,
    shared_library_list: List[str],
):
    if binary_file is None:
        raise Exception(f"{merged_file} doesn't have corresponding binary!")
    print("start to export: ", merged_file)
    # run export
    cmd_shared_library = (
        ""
        if not shared_library_list
        else f" -object  {' -object '.join(shared_library_list)}"
    )
    # if binary_file = "", then no need to add it (python test)
    cmd_binary = "" if not binary_file else f" -object {binary_file} "
    llvm_tool_path = utils.get_llvm_tool_path()
    cmd = f"{llvm_tool_path}/llvm-cov export {cmd_binary} {cmd_shared_library}  -instr-profile={merged_file} > {json_file}"
    print(cmd)
    os.system(cmd)
    
def export(test_list):
    print("start export")
    # find all merged profile under merged_folder and sub-folders
    g = os.walk(MERGED_FOLDER_BASE_DIR)
    for path, dir_list, file_list in g:
        # create corresponding merged folder in [json folder] if not exists yet
        utils.create_corresponding_folder(
            path, MERGED_FOLDER_BASE_DIR, dir_list, JSON_FOLDER_BASE_DIR
        )
        # check if we can find merged profile under this path's folder
        for file_name in file_list:
            if file_name.endswith(".merged"):
                if not utils.related_to_test_list(file_name, test_list):
                    continue
                print(f"start export {file_name}")
                # merged file
                merged_file = os.path.join(path, file_name)
                # json file
                json_file_name = utils.replace_extension(file_name, ".json")
                json_file = os.path.join(
                    JSON_FOLDER_BASE_DIR,
                    utils.convert_to_relative_path(path, MERGED_FOLDER_BASE_DIR),
                    json_file_name,
                )
                # binary file and shared library
                binary_file = "" # FIXME
                shared_library_list = []
                    # if it is python test, no need to provide binary, shared library is enough
                    #binary_file = (
                    #    ""
                    #    if test_name.endswith(".py")
                    #    else get_oss_binary_file(test_name, TestType.CPP)
                    #)
                shared_library_list = utils.get_shared_library()
                export_target(
                    merged_file,
                    json_file,
                    binary_file,
                    shared_library_list,
                )
# Clang Json parser

class LlvmCoverageSegment(t.NamedTuple):
    line: int
    col: int
    segment_count: int
    has_count: int
    is_region_entry: int
    is_gap_entry: Optional[int]

    @property
    def has_coverage(self):
        return self.segment_count > 0

    @property
    def is_executable(self):
        return self.has_count > 0

    def get_coverage(
        self, prev_segment: "LlvmCoverageSegment"
    ):
        # Code adapted from testpilot.testinfra.runners.gtestcoveragerunner.py
        if not prev_segment.is_executable:
            return [], []

        # this segment ends at the line if col == 1
        # (so segment effectively ends on the line) and
        # line+1 if col is > 1 (so it touches at least some part of last line).
        end_of_segment = self.line if self.col == 1 else self.line + 1
        lines_range = list(range(prev_segment.line, end_of_segment))
        return (lines_range, []) if prev_segment.has_coverage else ([], lines_range)

def parse_segments(raw_segments: List[List[int]]):
    """
        Creates LlvmCoverageSegment from a list of lists in llvm export json.
        each segment is represented by 5-element array.
    """
    ret: List[LlvmCoverageSegment] = []
    for raw_segment in raw_segments:
        assert (
            len(raw_segment) == 5 or len(raw_segment) == 6
        ), "list is not compatible with llvmcom export:"
        " Expected to have 5 or 6 elements"
        if len(raw_segment) == 5:
            ret.append(
                LlvmCoverageSegment(
                    raw_segment[0],
                    raw_segment[1],
                    raw_segment[2],
                    raw_segment[3],
                    raw_segment[4],
                    None,
                )
            )
        else:
            ret.append(LlvmCoverageSegment(*raw_segment))

    return ret
 
class LlvmCoverageParser:
    """
        Accepts a parsed json produced by llvm-cov export -- typically,
        representing a single C++ test and produces a list
        of CoverageRecord(s).

    """

    def __init__(self, llvm_coverage: Dict[str, Any]) -> None:
        self._llvm_coverage = llvm_coverage

    @staticmethod
    def _skip_coverage(path: str) -> bool:
        """
            Returns True if file path should not be processed.
            This is repo-specific and only makes sense for the current state of
            ovrsource.
        """
        if "/third-party/" in path:
            return True
        return False

    @staticmethod
    def _collect_coverage(
        segments: List[LlvmCoverageSegment],
    ) -> Tuple[List[int], List[int]]:
        """
            Stateful parsing of coverage segments.
        """
        covered_lines: Set[int] = set()
        uncovered_lines: Set[int] = set()
        prev_segment = LlvmCoverageSegment(1, 0, 0, 0, 0, None)
        for segment in segments:
            covered_range, uncovered_range = segment.get_coverage(prev_segment)
            covered_lines.update(covered_range)
            uncovered_lines.update(uncovered_range)
            prev_segment = segment

        uncovered_lines.difference_update(covered_lines)
        return sorted(covered_lines), sorted(uncovered_lines)

    def parse(self, repo_name: str):
        # The JSON format is described in the LLVM source code
        # https://github.com/llvm-mirror/llvm/blob/master/tools/llvm-cov/CoverageExporterJson.cpp
        records: List[CoverageRecord] = []
        for export_unit in self._llvm_coverage["data"]:
            for file_info in export_unit["files"]:
                filepath = file_info["filename"]
                if self._skip_coverage(filepath):
                    continue

                if filepath is None:
                    continue

                segments = file_info["segments"]

                covered_lines, uncovered_lines = self._collect_coverage(
                    parse_segments(segments)
                )

                records.append(CoverageRecord(filepath, covered_lines, uncovered_lines))

        return records
    