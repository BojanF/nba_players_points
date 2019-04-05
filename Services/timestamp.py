import datetime


def start_timestamp():
    return 'Start: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def end_timestamp():
    return 'End: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def start_timestamp_filename_w():
    return datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')


def end_timestamp_filename_w():
    return datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
