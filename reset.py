import os
if os.path.exists("coordinator_locks.db"):
    os.remove("coordinator_locks.db")
    print("Deleted coordinator_locks.db")
else:
    print("File not found — already clean")