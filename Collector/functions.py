from libs_import import *

def get_id(url):
    u_pars = urlparse(url)
    quer_v = parse_qs(u_pars.query).get('v')
    if quer_v:
        return quer_v[0]
    pth = u_pars.path.split('/')
    if pth:
        return pth[-1]


def build_service():
    """
    Create connection with YouTube API
    """
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    return build(YOUTUBE_API_SERVICE_NAME,
                 YOUTUBE_API_VERSION,
                 developerKey=key)


def remove_brackets(x):
    """
    Simplify response string
    """
    nstring = str(x)
    beginning_bracket = re.sub(r"'items': \[{", "'items' : {", nstring)
    ending_bracket = re.sub(r"}], 'pageInfo'", "}, 'pageInfo'", beginning_bracket)
    response_d = ast.literal_eval(ending_bracket)
    return response_d


def collect_data(response_d):
    """
    Construct data into structured dictionary
    """
    video_data = {'id': response_d['id']}
    video_data.update(response_d['statistics'])
    video_data.update(format_duration(response_d['contentDetails']))

    snippet = response_d['snippet']
    del snippet['localized']

    video_data.update(snippet)
    video_data.update({'date': date.today()})

    category_dict = {}
    for elem in list_of_category_info:
        category_dict[elem['id']] = elem['snippet']['title']

    for item in category_dict:
        if item == video_data['categoryId']:
            video_data['categoryId'] = category_dict[item]

    video_data_entries.append(category_dict)

    return video_data_entries


def format_duration(content_deets):
    """
    Format the duration of the video into seconds
    """
    sec_patrn = re.compile(r'(\d+)S')
    min_patrn = re.compile(r'(\d+)M')
    hr_patrn = re.compile(r'(\d+)H')

    for item in content_deets:
        duration = content_deets['duration']

        seconds = sec_patrn.search(duration)
        minutes = min_patrn.search(duration)
        hours = hr_patrn.search(duration)

        seconds = int(seconds.group(1)) if seconds else 0
        minutes = int(minutes.group(1)) if minutes else 0
        hours = int(hours.group(1)) if hours else 0

        vid_seconds = timedelta(
            hours=hours,
            minutes=minutes,
            seconds=seconds
        ).total_seconds()

        content_deets['duration'] = vid_seconds

        return content_deets


def export(video_data_entries, filename, columns):
    """
    Save data in *filename* csv table
    """
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns, restval=np.nan, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(video_data_entries)
