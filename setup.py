from setuptools import setup, find_packages

setup(
    name="soulillusions-agent",
    version="1.0.0",
    description="SoulIllusions — Autonomous AI Agent with Web UI, Book Writer, and Audiobook System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(),
    package_data={"soulillusions": ["webui/*.html", "default_config.json"]},
    include_package_data=True,
    install_requires=[
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "tts": ["pyttsx3>=2.90"],
        "dev": ["pytest>=7.0", "pytest-asyncio>=0.21"],
    },
    entry_points={
        "console_scripts": [
            "soulillusions=soulillusions.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
