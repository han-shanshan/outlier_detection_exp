import io
import os
import platform

from setuptools import setup, find_packages

try:
    # from wheel.bdist_wheel import bdist_wheel
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            self.root_is_pure = False
            self.universal = True
            _bdist_wheel.finalize_options(self)

except ImportError:
    bdist_wheel = None

requirements = [
    'GPUtil',
    'PyYAML',
    'aiohttp>=3.8.1',
    'attrdict',
    'attrs',
    'boto3',
    'cachetools',
    'chardet',
    'click',
    'dill',
    'docker==6.1.3',
    'fastapi',
    'gensim',
    'geventhttpclient>=1.4.4,<=2.0.9',
    'graphviz<0.9.0,>=0.8.1',
    'h5py',
    'httpx',
    'matplotlib',
    'multiprocess',
    'networkx<3.0',
    'ntplib',
    'numpy>=1.21',
    'onnx',
    'paho-mqtt<2.0.0',
    'pandas',
    'prettytable',
    'py-machineid',
    'pydantic',
    'pydantic-settings',
    'pytest',
    'pytest-mock',
    'python-rapidjson>=0.9.1',
    'redis',
    'scikit-learn',
    'smart-open==6.3.0',
    'spacy',
    'sqlalchemy',
    'toposort',
    'torch>=1.13.1',
    'torchvision>=0.14.1',
    'tqdm',
    'tritonclient',
    'typing_extensions',
    'tzlocal',
    'uvicorn',
    'wandb==0.13.2',
    'wget',
]

requirements_extra_mpi = [
    "mpi4py",
]

requirements_extra_tf = [
    "tensorflow",
    "tensorflow_datasets",
    "tensorflow_federated",
]

requirements_extra_jax = [
]

# https://github.com/apache/incubator-mxnet/issues/18329
requirements_extra_mxnet = [
    "mxnet==2.0.0b1",
]

requirements_extra_crypto = [
    "PyNaCl",
    "eciespy",
]

requirements_extra_fhe = [
    "tenseal",
]

requirements_extra_llm = [
    'accelerate>=0.24.0',
    'datasets>=2.14.0',
    'einops',
    'evaluate',
    'ninja',
    'packaging',
    'peft>=0.4.0',
    'safetensors',
    'sentencepiece',
    'transformers[torch]>=4.31.0',
    'zstandard'
]

requirements_extra_deepspeed = [
    "deepspeed>=0.10.2",
]

# if platform.machine() == "x86_64":
#    requirements.append("MNN==1.1.6")

setup(
    name="fedml",
    version="0.8.30",
    author="FederatedLearning Team",
    author_email="ch@fedml.ai",
    description="A research and production integrated edge-cloud library for "
                "federated/distributed machine learning at anywhere at any scale.",
    long_description=io.open(os.path.join("README.md"), "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/FederatedLearning-AI/FederatedLearning",
    keywords=[
        "distributed machine learning",
        "federated learning",
        "natural language processing",
        "computer vision",
        "Internet of Things",
    ],
    classifiers=[
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    packages=find_packages(),
    include_package_data=True,
    data_files=[
        (
            "fedml",
            [
                "fedml/config/simulation_sp/fedml_config.yaml",
                "fedml/config/simulaton_mpi/fedml_config.yaml",
                "fedml/computing/scheduler/build-package/mlops-core/fedml-server/server-package/conf/fedml.yaml",
                "fedml/computing/scheduler/build-package/mlops-core/fedml-server/server-package/fedml/config/fedml_config.yaml",
                "fedml/computing/scheduler/build-package/mlops-core/fedml-client/client-package/conf/fedml.yaml",
                "fedml/computing/scheduler/build-package/mlops-core/fedml-client/client-package/fedml/config/fedml_config.yaml",
                "fedml/computing/scheduler/master/templates/fedml-aggregator-data-pv.yaml",
                "fedml/computing/scheduler/master/templates/fedml-aggregator-data-pvc.yaml",
                "fedml/computing/scheduler/master/templates/fedml-server-deployment.yaml",
                "fedml/computing/scheduler/master/templates/fedml-server-svc.yaml",
                "fedml/core/mlops/ssl/open-dev.fedml.ai_bundle.crt",
                "fedml/core/mlops/ssl/open-test.fedml.ai_bundle.crt",
                "fedml/core/mlops/ssl/open-release.fedml.ai_bundle.crt",
                "fedml/core/mlops/ssl/open-root-ca.crt",
            ],
        )
    ],
    install_requires=requirements,
    extras_require={
        "MPI": requirements_extra_mpi,
        "deepspeed": requirements_extra_deepspeed,
        "fhe": requirements_extra_fhe,
        "gRPC": "grpcio",
        "jax": requirements_extra_jax,
        "llm": requirements_extra_llm,
        "mxnet": requirements_extra_mxnet,
        "tensorflow": requirements_extra_tf,
    },
    package_data={"": ["py.typed"]},
    license="Apache 2.0",
    entry_points={
        "console_scripts": [
            "fedml=fedml.cli.cli:cli",
        ]
    },
    cmdclass={"bdist_wheel": bdist_wheel},
    # options={"bdist_wheel": {"universal": True}}
)
