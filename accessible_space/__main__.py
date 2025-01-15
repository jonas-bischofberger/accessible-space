import os
import sys


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "validation":
            import accessible_space.apps.validation
            accessible_space.apps.validation.main()
        elif sys.argv[1] == "test":
            import pytest
            pytest.main([
                os.path.abspath(os.path.dirname(__file__)),
                "--doctest-modules",
                # "--filterwarnings=ignore:Inferring attacking direction:UserWarning",
                # "--filterwarnings=ignore:Range of tracking Y coordinates:UserWarning",
                # "--filterwarnings=ignore:Range of tracking X coordinates:UserWarning",
            ])
        elif sys.argv[1] == "demo":
            import accessible_space.apps.readme
            accessible_space.apps.readme.main()
        else:
            raise ValueError(f"Invalid argument: {sys.argv[1]}. Available arguments: 'validation', 'test', 'demo'.")
    else:
        raise ValueError("No arguments provided. Available arguments: 'validation', 'test', 'demo'.")


if __name__ == '__main__':
    main()
