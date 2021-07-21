import time
import threading

if __name__ == "__main__":

    start_time = time.time()

    print( "threads:  ", threading.active_count() )

    time.sleep(5)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("--- %s minutes ---" % ( (time.time() - start_time)/60 ))
    print("--- %s hours ---" % ( (time.time() - start_time)/3600 ))