import datetime

FMT = '%Y-%m-%d %H:%M:%S'


def start_timestamp():
    return 'Start: ' + datetime.datetime.now().strftime(FMT)


def end_timestamp():
    return 'End: ' + datetime.datetime.now().strftime(FMT)


def start_timestamp_filename_w():
    return datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')


def end_timestamp_filename_w():
    return datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')


def interval_between_dates(start_date, end_date):
    start = start_date[-19:]
    end = end_date[-19:]
    t_delta = datetime.datetime.strptime(end, FMT) - datetime.datetime.strptime(start, FMT)
    return t_delta.__str__()
