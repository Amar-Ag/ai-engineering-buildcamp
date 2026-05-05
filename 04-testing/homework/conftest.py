from cost_tracker import reset_cost_file, display_total_usage

def pytest_sessionstart(session):
    reset_cost_file()

def pytest_sessionfinish(session, exitstatus):
    display_total_usage()