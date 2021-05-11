

INBOUND_BUCKET = 'aquabyte-images-raw'

def get_capture_keys(pen_id: int, start_date: str, end_date: str, inbound_bucket=INBOUND_BUCKET) -> List:
    """Take pen_id_time_range dataset as an input, and return list of paired_urls and corresponding
    crop_metadatas."""

    site_id = PEN_SITE_MAPPING[pen_id]
    dates = get_dates_in_range(start_date, end_date)
    capture_keys = []
    for date in dates:
        print('Getting capture keys for pen_id={}, date={}...'.format(pen_id, date))
        for hour in DAYTIME_HOURS_GMT:
            s3_prefix = 'environment=production/site-id={}/pen-id={}/date={}/hour={}'.format(site_id, pen_id,
                                                                                    date, hour)

            generator = s3.get_matching_s3_keys(inbound_bucket, prefix=s3_prefix,
                                                            subsample=1.0,
                                                            suffixes=['capture.json'])

            these_capture_keys = [key for key in generator]
            capture_keys.extend(these_capture_keys)

    return capture_keys



def get_image_urls(capture_keys):
    """Gets left urls, right urls, and crop metadatas corresponding to capture keys."""

    left_urls, crop_metadatas = [], []
    for capture_key in capture_keys:

        # get image URLs
        left_image_key = capture_key.replace('capture.json', 'left_frame.resize_512_512.jpg')
        left_image_url = os.path.join('s3://', INBOUND_BUCKET, left_image_key)
        left_urls.append(left_image_url)

    return left_urls


def process_into_plali_records(image_urls):

    values_to_insert = []
    for idx, image_url in enumerate(image_urls):
        id = str(uuid.uuid4())
        images = {image_url}
        metadata = {}
        priority = float(idx) / len(image_urls)

        values = {
            'id': id,
            'workflow_id': '00000000-0000-0000-0000-000000000047',
            'images': images,
            'metadata': metadata,
            'priority': priority
        }

        values_to_insert.append(values)

    return values_to_insert