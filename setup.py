from setuptools import setup, find_packages

setup(
    name="adaptive_clinical_trial",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy", "scipy", "pandas", "matplotlib",
        "torch", "scikit-learn", "tqdm"
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "run-all-scenarios = scripts.run_all_scenarios:main",
        ]
    },
)
