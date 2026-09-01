Installation
==============================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

You need the following to install OpenEquivariance:

- A Linux system equipped with an NVIDIA / AMD graphics card.
- Either PyTorch >= 2.4 (>= 2.8 for AOTI and export), or JAX >= 0.8.2
  with CUDA or RocM support. 
- GCC 9+ and the CUDA / HIP toolkit. The command
  ``c++ --version`` should return >= 9.0; see below for details on 
  setting an alternate compiler.

.. tab:: PyTorch

    Installation is one easy command, followed by import verification: 

    .. code-block:: bash 

        pip install openequivariance
        python -c "import openequivariance"

    The second line triggers a build of the C++ extension we use to compile
    kernels, which can take a couple of minutes. Subsequent imports are
    much faster since this extension is cached.

    To support ``torch.compile``, ``torch.export``, and
    JITScript, OpenEquivariance needs to compile a C++ extension
    tightly integrated with PyTorch. If you see a warning that
    this extension could not be compiled, first check:

    .. code-block:: bash 

        c++ --version 
        
    To build the extension with an alternate compiler, set the 
    ``CC`` and ``CXX``
    environment variable and retry the import:

    .. code-block:: bash

        export CC=/path/to/your/gcc
        export CXX=/path/to/your/g++
        python -c "import openequivariance"


    These configuration steps are required only ONCE after 
    installation (or upgrade) with pip. 


.. tab:: JAX NVIDIA GPUs 

    First ensure the appropriate JAX Python
    package is installed in your environment. Then
    run the following two commands stricly in order:

    .. code-block:: bash 

        pip install openequivariance[jax]
        pip install openequivariance_extjax --no-build-isolation

.. tab:: JAX AMD GPUs 

    Ensure that JAX is installed correctly with RocM support
    before running, in order,

    .. code-block:: bash 

        pip install openequivariance[jax]
        JAX_HIP=1 pip install openequivariance_extjax --no-build-isolation


.. tab:: Nightly (PT) 

    .. code-block:: bash 

        pip install "git+https://github.com/PASSIONLab/OpenEquivariance#subdirectory=openequivariance"


.. tab:: Nightly (JAX)

    .. code-block:: bash 

        pip install "git+https://github.com/PASSIONLab/OpenEquivariance#subdirectory=openequivariance[jax]"
        pip install "git+https://github.com/PASSIONLab/OpenEquivariance#subdirectory=openequivariance_extjax --no-build-isolation"

        # Use the command below for JAX+AMD
        # JAX_HIP=1 pip install "git+https://github.com/PASSIONLab/OpenEquivariance#subdirectory=openequivariance_extjax --no-build-isolation"


If you're using JAX, set the environment variable 
``OEQ_NOTORCH=1`` to avoid a PyTorch import: 

.. code-block:: bash

    export OEQ_NOTORCH=1
    python -c "import openequivariance.jax"

JAX kernel compilation
----------------------

The CUDA JAX extension compiles generated kernels in parallel. Configure it
with the following environment variables before importing OpenEquivariance:

``OEQ_JAX_COMPILER_THREADS``
    Number of kernels compiled at the same time. The valid range is 1 to 64,
    and the default is 8.

``OEQ_JAX_COMPILER_QUEUE_CAPACITY``
    Number of compilation jobs that may wait for a worker. The valid range is
    1 to 256, and the default is 32. Submission waits when the queue is full.

For example, this configuration compiles up to 16 kernels at once:

.. code-block:: bash

    export OEQ_JAX_COMPILER_THREADS=16
    python application.py

These variables are read once when kernel compilation first starts. The HIP
path compiles synchronously and does not use these settings.


Configurations on Major Platforms 
---------------------------------
OpenEquivariance has been tested on both supercomputers and lab clusters.
Here are some tested environment configuration files. If you use OpenEquivariance
on a major cluster, send us a pull request to add your configuration! 


.. tab:: NERSC Perlmutter (NVIDIA A100) 

    .. code-block:: bash
        :caption: env.sh (last updated June 2025)

        module load gcc 
        module load conda

        # Deactivate any base environments
        for i in $(seq ${CONDA_SHLVL}); do 
            conda deactivate
        done

        conda activate <your-conda-env>


.. tab:: OLCF Frontier (AMD MI250x)

    You need to install a HIP-enabled verison of PyTorch to use our package. 
    Follow the steps `here <https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html>`_.


    .. code-block:: bash
        :caption: env.sh (last updated June 2025) 

        module load PrgEnv-gnu/8.6.0
        module load miniforge3/23.11.0-0
        module load rocm/6.4.0
        module load craype-accel-amd-gfx90a

        for i in $(seq ${CONDA_SHLVL}); do
            conda deactivate
        done

        conda activate <your-conda-env>
        export CC=cc
        export CXX=CC
