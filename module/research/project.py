import re
import os

from datetime import datetime, timedelta
from scipy import signal

from module.base.utils import *
from module.logger import logger
from module.map_detection.utils import rgb2gray
from module.ocr.ocr import Ocr
from module.research.assets import *
from module.research.filter import Filter
from module.research.preset import *
from module.research.project_data import LIST_RESEARCH_PROJECT
from module.ui.ui import UI

from PIL import Image

RESEARCH_SERIES = [SERIES_1, SERIES_2, SERIES_3, SERIES_4, SERIES_5]
OCR_RESEARCH = [OCR_RESEARCH_1, OCR_RESEARCH_2, OCR_RESEARCH_3, OCR_RESEARCH_4, OCR_RESEARCH_5]
OCR_RESEARCH = Ocr(OCR_RESEARCH, name='RESEARCH', threshold=64, alphabet='0123456789BCDEGHQTMIULRF-')
FILTER_REGEX = re.compile('(s[123])?'
                          '-?'
                          '(neptune|monarch|ibuki|izumo|roon|saintlouis|seattle|georgia|kitakaze|azuma|friedrich|gascogne|champagne|cheshire|drake|mainz|odin)?'
                          '(dr|pry)?'
                          '([bcdeghqt])?'
                          '-?'
                          '(\d.\d|\d\d?)?')
FILTER_ATTR = ('series', 'ship', 'ship_rarity', 'genre', 'duration')
FILTER_PRESET = ('shortest', 'cheapest', 'reset')
FILTER = Filter(FILTER_REGEX, FILTER_ATTR, FILTER_PRESET)


def get_research_series(image):
    """
    Get research series using a simple color detection.
    May not be able to detect 'IV' and 'V' in the future research series.

    Args:
        image (PIL.Image.Image):

    Returns:
        list[int]: Such as [1, 1, 1, 2, 3]
    """
    result = []
    parameters = {'height': 200}

    for button in RESEARCH_SERIES:
        im = np.array(image.crop(button.area).resize((46, 25)).convert('L'))
        mid = np.mean(im[8:17, :], axis=0)
        peaks, _ = signal.find_peaks(mid, **parameters)
        series = len(peaks)
        if 1 <= series <= 3:
            result.append(series)
        else:
            result.append(0)
            logger.warning(f'Unknown research series: button={button}, series={series}')

    return result


def get_research_name(image):
    """
    Args:
        image (PIL.Image.Image):

    Returns:
        list[str]: Such as ['D-057-UL', 'D-057-UL', 'D-057-UL', 'D-057-UL', 'D-057-UL']
    """
    names = []
    for name in OCR_RESEARCH.ocr(image):
        # S3 D-022-MI (S3-Drake-0.5) detected as 'D-022-ML', because of Drake's white cloth.
        name = name.replace('ML', 'MI').replace('MIL', 'MI')
        names.append(name)
    return names


class ResearchProject:
    REGEX_SHIP = re.compile(
        '(neptune|monarch|ibuki|izumo|roon|saintlouis|seattle|georgia|kitakaze|azuma|friedrich|gascogne|champagne|cheshire|drake|mainz|odin)')
    REGEX_INPUT = re.compile('(coin|cube|part)')
    DR_SHIP = ['azuma', 'friedrich', 'drake']

    def __init__(self, name, series):
        """
        Args:
            name (str): Such as 'D-057-UL'
            series (int): Such as 1, 2, 3
        """
        self.valid = True
        # self.config = config
        self.name = self.check_name(name)
        self.series = f'S{series}'
        self.genre = ''
        self.duration = '24'
        self.ship = ''
        self.ship_rarity = ''
        self.need_coin = False
        self.need_cube = False
        self.need_part = False

        matched = False
        for data in self.get_data(name=self.name, series=series):
            matched = True
            self.data = data
            self.genre = data['name'][0]
            self.duration = str(data['time'] / 3600).rstrip('.0')
            for item in data['input']:
                result = re.search(self.REGEX_INPUT, item['name'].replace(' ', '').lower())
                if result:
                    self.__setattr__(f'need_{result.group(1)}', True)
            for item in data['output']:
                result = re.search(self.REGEX_SHIP, item['name'].replace(' ', '').lower())
                if not self.ship:
                    self.ship = result.group(1) if result else ''
                if self.ship:
                    self.ship_rarity = 'dr' if self.ship in self.DR_SHIP else 'pry'
            break

        if not matched:
            logger.warning(f'Invalid research {self}')
            self.valid = False

    def __str__(self):
        if self.valid:
            return f'{self.series} {self.name}'
        else:
            return f'{self.series} {self.name} (Invalid)'

    @staticmethod
    def check_name(name):
        """
        Args:
            name (str):

        Returns:
            str:
        """
        name = name.strip('-')
        parts = name.split('-')
        if len(parts) == 3:
            prefix, number, suffix = parts
            number = number.replace('D', '0').replace('O', '0').replace('S', '5')
            return '-'.join([prefix, number, suffix])
        return name

    def get_data(self, name, series):
        """
        Args:
            name (str): Such as 'D-057-UL'
            series (int): Such as 1, 2, 3

        Yields:
            dict:
        """
        for data in LIST_RESEARCH_PROJECT:
            if (data['series'] == series) and (data['name'] == name):
                yield data

        if name.startswith('D'):
            # Letter 'C' may recognized as 'D', because project card is shining.
            name1 = 'C' + self.name[1:]
            for data in LIST_RESEARCH_PROJECT:
                if (data['series'] == series) and (data['name'] == name1):
                    self.name = name1
                    yield data

        for data in LIST_RESEARCH_PROJECT:
            if (data['series'] == series) and (data['name'].rstrip('MIRFUL-') == name.rstrip('MIRFUL-')):
                yield data

        return False


class ResearchSelector(UI):
    projects: list

    def research_detect(self, image):
        """
        Args:
            image (PIL.Image.Image): Screenshots
        """
        projects = []
        for name, series in zip(get_research_name(image), get_research_series(image)):
            project = ResearchProject(name=name, series=series)
            logger.attr('Project', project)
            projects.append(project)

        self.projects = projects

    def research_sort_filter(self):
        """
        Returns:
            list: A list of str and int, such as [2, 3, 0, 'reset']
        """
        # Load filter string
        preset = self.config.RESEARCH_FILTER_PRESET
        if preset == 'customized':
            string = self.config.RESEARCH_FILTER_STRING
        else:
            if preset not in DICT_FILTER_PRESET:
                logger.warning(f'Preset not found: {preset}, use default preset')
                preset = 'series_3_than_2'
            string = DICT_FILTER_PRESET[preset]

        FILTER.load(string)
        priority = FILTER.apply(self.projects)
        priority = self._research_check_filter(priority)

        # Log
        logger.attr(
            'Filter_sort',
            ' > '.join([self.projects[index].name if isinstance(index, int) else index for index in priority]))
        return priority

    def _research_check_filter(self, priority):
        """
        Args:
            priority (list): A list of str and int, such as [2, 3, 0, 'reset']

        Returns:
            list: A list of str and int, such as [2, 3, 0, 'reset']
        """
        out = []
        for index in priority:
            if isinstance(index, str):
                out.append(index)
                continue
            proj = self.projects[index]
            if not proj.valid:
                continue
            if (not self.config.RESEARCH_USE_CUBE and proj.need_cube) \
                    or (not self.config.RESEARCH_USE_COIN and proj.need_coin) \
                    or (not self.config.RESEARCH_USE_PART and proj.need_part):
                continue
            # Reasons to ignore B series and E-2:
            # - Can't guarantee research condition satisfied.
            #   You may get nothing after a day of running, because you didn't complete the precondition.
            # - Low income from B series research.
            #   Gold B-4 basically equivalent to C-12, but needs a lot of oil.
            if (proj.genre.upper() == 'B') \
                    or (proj.genre.upper() == 'E' and str(proj.duration) != '6'):
                continue
            out.append(index)
        return out

    def research_sort_shortest(self):
        """
        Returns:
            list: A list of str and int, such as [2, 3, 0, 'reset']
        """
        FILTER.load(FILTER_STRING_SHORTEST)
        priority = FILTER.apply(self.projects)
        priority = self._research_check_filter(priority)

        logger.attr(
            'Shortest_sort',
            ' > '.join([self.projects[index].name if isinstance(index, int) else index for index in priority]))
        return priority

    def research_sort_cheapest(self):
        """
        Returns:
            list: A list of str and int, such as [2, 3, 0, 'reset']
        """
        FILTER.load(FILTER_STRING_CHEAPEST)
        priority = FILTER.apply(self.projects)
        priority = self._research_check_filter(priority)

        logger.attr(
            'Cheapest_sort',
            ' > '.join([self.projects[index].name if isinstance(index, int) else index for index in priority]))
        return priority

    def research_detect_jp(self, images):
        """
        Args:
            images (list of PIL.Image.Image): Screenshots
        """
        projects = []
        for image in images:
            project = ResearchProjectJp(image = image)
            logger.attr('Project', project.name)
            projects.append(project)

        self.projects = projects


class ResearchProjectJp:
    REGEX_SHIP = re.compile(
        '(neptune|monarch|ibuki|izumo|roon|saintlouis|seattle|georgia|kitakaze|azuma|friedrich|gascogne|champagne|cheshire|drake|mainz|odin)')
    REGEX_INPUT = re.compile('(coin|cube|part)')
    DR_SHIP = ['azuma', 'friedrich', 'drake']

    def __init__(self, image):
        self.valid = True

        self.name = ''
        self.image = image
        self.series = ''
        self.genre = ''
        self.duration = '24'
        self.ship = ''
        self.ship_rarity = ''

        #Todo: add need_coin, need_cube, need_part check.
        self.need_coin = False
        self.need_cube = False
        self.need_part = False

        self.get_research_series_jp()
        self.get_research_duration_jp()
        self.get_research_genre_jp()
        
        if (self.genre == "D"):
            self.get_research_ship_jp()
        if self.ship:
            self.ship_rarity = 'dr' if self.ship in self.DR_SHIP else 'pry'
        self.name = self.series + '-' + self.genre + '-' + self.duration + ' ' + self.ship
        logger.info(f'{self.series} {self.genre} {self.duration} {self.ship} {self.ship_rarity}')
            
    def get_research_series_jp(self):
        parameters = {'height': 200}
        area = (285, 109, 319, 134)
        button = Button(area=area, color=(), button=area, name='SERIES_DETAIL')
        im = np.array(self.image.crop(button.area).resize((46, 25)).convert('L'))
        mid = np.mean(im[8:17, :], axis=0)
        peaks, _ = signal.find_peaks(mid, **parameters)
        series = len(peaks)
        if 1 <= series <= 3:
            self.series = f'S{series}'
        else:
            self.series = ''
            self.valid = False
        logger.info(f'Series: {self.series}')

    def get_research_duration_jp(self):
        area = (790, 275, 911, 321)
        button = Button(area=area, color=(), button=area, name='DURATION')
        ocr = Ocr(button, alphabet='0123456789:')
        self.duration = str(self._parse_time(ocr.ocr(self.image)).total_seconds() / 3600).rstrip('.0')
        logger.info(f'Duration: {self.duration}')
        
    def get_research_genre_jp(self):
        '''
        Looks ugly and unstable (should allow some offsets).
        Maybe should change to use Button.match() instead of MatchTemplate class.
        '''
        area = (323, 110, 418, 133)
        button = Button(area=area, color=(), button=area, name='GENRE_DETAIL')
        genre_folder = './assets/jp/research/research_genre'
        im = self.image.crop(button.area)
        ship_templates = MatchTemplate(folder = genre_folder)
        self.genre = ship_templates.match_template(im)
    
    def get_research_ship_jp(self):
        '''
        2.5, 5, and 8 hours' D research have 4 items, while 0.5 hours' one has 3.
        Need to check this to save ship templetes.
        Can remove this check and simply add some offsets while matching once the collection is finished.
        '''
        area_even = (331, 448, 407, 524)
        area_odd = (377, 448, 453, 524)
        area = area_odd if self._stats_items_num_is_odd(self.image) else area_even
        ship_folder = './assets/jp/research/research_ship'
        im = self.image.crop(area)
        #Save ship templetes.
        image_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        im.save(f'./log/ship/{image_time}.png')
        
        ship_templates = MatchTemplate(folder = ship_folder)
        self.ship = ship_templates.match_template(im)

    @staticmethod
    def _parse_time(string):
        """
        Args:
            string (str): Such as 01:00:00, 05:47:10, 17:50:51.
        Returns:
            timedelta: datetime.timedelta instance.
        """
        result = re.search('(\d+):(\d+):(\d+)', string)
        if not result:
            logger.warning(f'Invalid time string: {string}')
            return None
        else:
            result = [int(s) for s in result.groups()]
            return timedelta(hours=result[0], minutes=result[1], seconds=result[2])

    @staticmethod
    def _stats_items_num_is_odd(image):
        """
        Args:
            image: Pillow image
        Returns:
            bool: If the number of items in row is odd.
        """
        image = np.array(image.crop(DETAIL_ITEMS_ODD.area))
        # Item pictures have a much high standard deviation than backgrounds.
        return np.std(rgb2gray(image)) > 10

class MatchTemplate:
    
    similarity = 0.85
    
    def __init__(self, folder):
        """
        Args:
            folder (str): Template folder.
        """
        self.folder = folder
        self.templates = {}
        self.load_template_folder()

    def load_template_folder(self):
        """
        Args:
            folder (str): Template folder.
        """
        data = self._load_folder(self.folder)
        for name, image in data.items():
            if name in self.templates:
                continue
            image = self._load_image(image)
            self.templates[name] = np.array(image)

    def match_template(self, image):
        """
        Haven't collected all ship templates.
        Templates in use now are resized from bigger images and may not match perfectly, 
        so we compare all results and choose the most similar one.
        It is not a good idea to simply set a threshold,
        for some templates have high similarity (especially kitakaze and seattle).
        Args:
            image: Image to match.
        Returns:
            str: Template name.
        """
        image = np.array(image)
        names = np.array(list(self.templates.keys()))[::]
        similarity = self.similarity
        result = ''
        for name in names:
            res = cv2.matchTemplate(image, self.templates[name], cv2.TM_CCOEFF_NORMED)
            _, sim, _, _ = cv2.minMaxLoc(res)
            logger.info(f'{name} {sim}')
            if sim > similarity:
                similarity = sim
                result = name
        logger.info(f'Matched template: {result}')
        return result

    @staticmethod
    def _load_folder(folder):
        """
        Args:
            folder (str): Template folder contains images.
        Returns:
            dict: Key: str, image file base name. Value: full filepath.
        """
        if not os.path.exists(folder):
            return {}

        out = {}
        for file in os.listdir(folder):
            name = os.path.splitext(file)[0]
            out[name] = os.path.join(folder, file)
    
        return out
    
    @staticmethod
    def _load_image(file):
        """
        Args:
            file (str): Path to file.
        Returns:
            Pillow image.
        """
        return Image.open(file).convert('RGB')

#def main():
#    image1 = Image.open('./test.png').convert('RGB')
#    test = ResearchProjectJp(image = image1)

#main()
