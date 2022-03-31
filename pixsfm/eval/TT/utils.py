from pathlib import Path
import subprocess
from dataclasses import dataclass, fields

DATASET_PATH = Path("./datasets/TT/")
OUTPUTS_PATH = Path("./outputs/TT/")

TRAINING = [
    "Barn",
    "Caterpillar",
    "Church",
    "Courthouse",
    "Ignatius",
    "Meetingroom",
    "Truck",
]
INTERMEDIATE = [
    "Family",
    "Francis",
    "Horse",
    "Lighthouse",
    "M60",
    "Panther",
    "Playground",
    "Train",
]

SCENES_TAU = {
    "Barn": 0.01,
    "Caterpillar": 0.005,
    "Church": 0.025,
    "Courthouse": 0.025,
    "Ignatius": 0.003,
    "Meetingroom": 0.01,
    "Truck": 0.005,
}


@dataclass
class Paths:
    colmap_exe: str = 'colmap'

    image_dir: Path = '{dataset}/image_sets/{scene}/'
    output_dir: Path = '{outputs}/{scene}/{tag}/'

    # colmap
    database: Path = '{outputs}/{scene}/database.db'
    database_refined: Path = '{outputs}/{scene}/{tag}/refined.db'

    # hloc
    features: Path = '{outputs}/{scene}/features_{method}.h5'
    matches: Path = '{outputs}/{scene}/matches_{method}.h5'
    retrieval: Path = '{outputs}/{scene}/features_retrieval.h5'
    pairs: Path = '{outputs}/{scene}/pairs.txt'

    sfm: Path = '{outputs}/{scene}/{tag}/sparse/'
    sfm_refined: Path = '{outputs}/{scene}/{tag}/sparse/refined/'
    mvs: Path = '{outputs}/{scene}/{tag}/dense/'
    pointcloud: Path = '{outputs}/{scene}/{tag}/dense/fused.ply'
    trajectory: Path = '{outputs}/{scene}/{tag}/cameras.log'

    gt_dir: Path = '{dataset}/trainingdata/{scene}/'
    eval_dir: Path = '{outputs}/{scene}/{tag}/evaluation/'
    eval_tool: Path = 'third-party/TanksAndTemples/python_toolbox/evaluation/'

    def interpolate(self, **kwargs):
        args = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None:
                val = str(val)
                for k, v in kwargs.items():
                    val = val.replace(f'{{{k}}}', str(v))
            args[f.name] = f.type(val)
        return self.__class__(**args)


class CalledProcessError(subprocess.CalledProcessError):
    def __str__(self):
        message = "Command '%s' returned non-zero exit status %d." % (
                self.cmd, self.returncode)
        if self.output is not None:
            message += ' Last outputs:\n%s' % (
                '\n'.join(self.output.decode('utf-8').split('\n')[-10:]))
        return message


def run_command(cmd, verbose=False):
    stdout = None if verbose else subprocess.PIPE
    ret = subprocess.run(
        cmd, stderr=subprocess.STDOUT, stdout=stdout, shell=True)
    if not ret.returncode == 0:
        raise CalledProcessError(
                returncode=ret.returncode, cmd=cmd, output=ret.stdout)
