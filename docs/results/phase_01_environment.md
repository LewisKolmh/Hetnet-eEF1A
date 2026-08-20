# Phase 1: Environment results

## Outputs produced

- VS Code project workspace
- Git repository
- Conda environment named `eef1a-hetnet`
- Python 3.10 environment
- Installed `hetnetpy` and `hetmatpy`
- Environment validation script
- Package-version records

## Quality-control results

The environment validation script completed successfully.

The active Python executable was located inside:

`miniconda3/envs/eef1a-hetnet/bin/python`

All required packages imported successfully.

## Errors or unresolved issues

The VS Code terminal initially opened one directory above the project root.
This caused Python to search for the validation script in the wrong
location. Changing into the `eef1a-hetnet` directory resolved the issue.

## Decision

The software environment passed validation. The project can proceed to
construction of a manually verifiable toy heterogeneous network.