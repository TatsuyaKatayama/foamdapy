import shutil
import shlex
import numpy as np
import os

from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Execution.BasicRunner import BasicRunner

from .tools import extract_val
from .tools import update_of


class OFCase:
    def __init__(self, case_dir):
        self.case_dir = case_dir

    def getValues(self, time_name: str, x_names: list, cells: list = None):
        return extract_val(self.case_dir, time_name, x_names, cells)

    def writeValues(self, x_data: np.ndarray, time_name: str, x_names: list):
        update_of(x_data, self.case_dir, time_name, x_names)

    def clearLogs(self):
        self.clearPattern("PyFoam*")
        self.clearPattern("log.*")

    def clearPattern(self, pattern: str):
        case = SolutionDirectory(self.case_dir)
        # PyFoamの出力とlogを削除
        case.clearPattern(pattern)

    def copyTimeDir(self, time_name: str, to_time_name: str, dirs_exist_ok=False):
        fromDir = os.path.join(self.case_dir, time_name)
        toDir = os.path.join(self.case_dir, to_time_name)
        new_path = shutil.copytree(fromDir, toDir, dirs_exist_ok)
        return new_path

    def forcast(self, end_time: str, writeInterval: str = None):
        # 再計算しない
        time_dir = os.path.join(self.case_dir, end_time)
        if os.path.exists(time_dir):
            return

        # Logの削除
        self.clearLogs()

        # controlDictの編集
        controlDict = ParsedParameterFile(
            os.path.join(self.case_dir, "system/controlDict")
        )
        controlDict.content["startTime"] = 0
        controlDict.content["startFrom"] = "latestTime"
        controlDict.content["endTime"] = end_time
        if writeInterval is not None:
            controlDict.content["writeInterval"] = writeInterval

        app = controlDict.content["application"]
        controlDict.writeFile()

        # "xxFoam -case path_to_case"コマンドを分割
        foamCMD = shlex.split(f"{app} -case {self.case_dir}")
        # xxFoamを実行を行うインスタンスの生成
        foamRunner = BasicRunner(foamCMD, silent=True)
        # xxFoamの実行。実行結果情報を返す
        foamState = foamRunner.start()
        return foamState
