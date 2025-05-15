import scrapy
from bs4 import BeautifulSoup

class LocationSpider(scrapy.Spider):
    name = 'locationSpider'
    start_urls = ['https://strangerthings.fandom.com/wiki/Category:Locations']

    def parse(self, response):
        for href in response.css('.category-page__members')[0].css("a::attr(href)").extract():
            extracted_data = scrapy.Request("https://strangerthings.fandom.com" + href,
                           callback=self.parse_location)
            yield extracted_data
            
    def parse_location(self, response):
        # get name 
        location_name = response.css("span.mw-page-title-main::text").extract_first(default='').strip()

        div_selector = response.css("div.mw-content-ltr.mw-parser-output").extract_first(default='')
        soup = BeautifulSoup(div_selector, 'html.parser').find('div')
        
        # get type
        location_type = ''
        if soup and soup.find('aside'):
            aside = soup.find('aside')
            for cell in aside.find_all('div', {'class': 'pi-data'}):
                if cell.find('h3'):
                    cell_name = cell.find('h3').text.strip()
                    if cell_name == 'Type': 
                        location_type = cell.find('div').text.strip()
        
        # get description
        if soup and soup.find('aside'):
            soup.find('aside').decompose()
        
        toc_div = soup.find('div', class_='toc')
        if not toc_div:
            toc_div = soup.find('table', class_='mw-collapsible')
        
        # extract content above the 'toc' div or 'table.mw-collapsible'
        location_description = ''
        for element in soup.find_all():
            if element == toc_div:
                break
            location_description += element.get_text(separator=' ', strip=True) + ' '
        
        location_description = location_description.strip()
        
        return dict(
            location_name=location_name,
            location_type=location_type,
            location_description=location_description
        )