from setuptools import setup, find_packages

setup(
    name="CMTarget-LLM",          # 项目名称
    version="0.1.0",              # 版本号
    author="HuanGu",              # 作者
    description="A brief description of your LLM project",
    packages=find_packages(),     # 自动寻找项目中的所有文件夹（需包含 __init__.py）
    install_requires=[            # 依赖列表，安装时会自动下载
        "torch>=2.0.0",
        "transformers>=4.57.6, <5.0.0",
        "numpy>=1.23.5,<2.0.0",
        "tqdm",
        "pandas>=2.3.3",
        "rdkit-pypi>=2022.9.5",
        "gensim>=4.4.0",
        "scikit-learn>=1.1.3",
        "matplotlib>=3.6.3",
        "peft>=0.17.1"
    ],
    python_requires=">=3.9",      # Python 版本要求
)