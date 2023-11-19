# ds-house-of-fragrance

## Scraper

### Scrapeops.io API
  ```python
      response = requests.get(
      url='https://proxy.scrapeops.io/v1/',
      params={
          'api_key': api_key,
          'url': url,
          'render_js': 'true',
      },
  )
  ```
'render_js' set to true for Java Rendering. 
Replace url and api_key

###
[output](./output.html): Example of the html get from product of fragrantica

[brand_json](./brand_json): urls of products in each brand

[brand_urls](./brand_urls.csv): table of urls of each brand

[scrap_product](./scrap_product.py): python code for scraping each product in [brand_json](./brand_json)
