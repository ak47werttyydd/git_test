import os
from enum import Enum
from typing import Dict, List, Set
import typing as t

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

class Test:
    name: str
    prefix: str
    
    def __init__(
        self, 
        name: str,
        prefix: str = "./"
    ):
        self.name = name
        self.prefix = prefix

# compiler type
class CompilerType(Enum):
    CLANG: str = "clang"
    GCC: str = "gcc"

class CoverageRecord(t.NamedTuple):
    filepath: str
    covered_lines: t.List[int]
    uncovered_lines: t.Optional[t.List[int]] = None

    def to_dict(self) -> t.Dict[str, t.Any]:
        return {
            "filepath": self.filepath,
            "covered_lines": self.covered_lines,
            "uncovered_lines": self.uncovered_lines,
        }
        
TestStatusType = Dict[str, Set[str]]
TestList = List[Test]