"""
Script to run all tests in the project.
"""

import os
import sys
import unittest
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_tests():
    """Run all tests in the project."""
    try:
        # Create test results directory
        os.makedirs("test_results", exist_ok=True)
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create test result file
        result_file = f"test_results/test_results_{timestamp}.txt"
        
        # Discover and run tests
        logger.info("Starting test discovery...")
        test_loader = unittest.TestLoader()
        test_suite = test_loader.discover('tests', pattern='test_*.py')
        
        # Run tests and capture results
        with open(result_file, 'w') as f:
            runner = unittest.TextTestRunner(stream=f, verbosity=2)
            result = runner.run(test_suite)
            
            # Write summary
            f.write("\nTest Summary:\n")
            f.write(f"Total tests: {result.testsRun}\n")
            f.write(f"Failures: {len(result.failures)}\n")
            f.write(f"Errors: {len(result.errors)}\n")
            f.write(f"Skipped: {len(result.skipped)}\n")
            
            # Write failures
            if result.failures:
                f.write("\nFailures:\n")
                for failure in result.failures:
                    f.write(f"{failure[0]}\n")
                    f.write(f"{failure[1]}\n")
            
            # Write errors
            if result.errors:
                f.write("\nErrors:\n")
                for error in result.errors:
                    f.write(f"{error[0]}\n")
                    f.write(f"{error[1]}\n")
        
        # Print summary to console
        logger.info(f"Test results written to {result_file}")
        logger.info(f"Total tests: {result.testsRun}")
        logger.info(f"Failures: {len(result.failures)}")
        logger.info(f"Errors: {len(result.errors)}")
        logger.info(f"Skipped: {len(result.skipped)}")
        
        # Return success if no failures or errors
        return len(result.failures) == 0 and len(result.errors) == 0
        
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 