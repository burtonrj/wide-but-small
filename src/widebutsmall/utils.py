from typing import Iterable
from typing import Union

from IPython import get_ipython
from tqdm import tqdm
from tqdm import tqdm_notebook


def which_environment() -> str:
    """
    Test if module is being executed in the Jupyter environment.

    Returns
    -------
    str
        'jupyter', 'ipython' or 'terminal'
    """
    try:
        ipy_str = str(type(get_ipython()))
        if "zmqshell" in ipy_str:
            return "jupyter"
        if "terminal" in ipy_str:
            return "ipython"
    except:
        return "terminal"


def progress_bar(x: Iterable, verbose: bool = True, **kwargs) -> Union[tqdm, Iterable]:
    """
    Generate a progress bar using the tqdm library. If execution environment is Jupyter, return tqdm_notebook
    otherwise used tqdm.

    Parameters
    -----------
    x: iterable
        some iterable to pass to tqdm function
    verbose: bool, (default=True)
        Provide feedback (if False, no progress bar produced)
    kwargs:
        additional keyword arguments for tqdm
    :return: tqdm or tqdm_notebook, depending on environment
    """
    if not verbose:
        return x
    if which_environment() == "jupyter":
        return tqdm_notebook(x, **kwargs)
    return tqdm(x, **kwargs)
