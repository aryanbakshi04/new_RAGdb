# This patches the sqlite3 module to use a newer version
import os
import sys
import platform

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
