import os
import subprocess
import time
from typing import List
import utils

import clang_coverage
from parser import summarize_jsons
from setting import (
    JSON_FOLDER_BASE_DIR,
    LOG_DIR,
    MERGED_FOLDER_BASE_DIR,
    PROFILE_DIR,
    SUMMARY_FOLDER_DIR,
    Test
)


if __name__ == "__main__":
    utils.create_folders()
    raw_file = os.path.join(utils.get_raw_profiles_folder(), "bug4.py" + ".profraw")
    test_list = [Test("bug4.py")]
    binary_file = utils.get_binary_file(test_list[0])
    clang_coverage.run_target(binary_file, raw_file)
    clang_coverage.merge(test_list)
    clang_coverage.export(test_list)
    test_list = [Test("bug4.py")]
    summarize_jsons(test_list, ["src"], [""])
    


