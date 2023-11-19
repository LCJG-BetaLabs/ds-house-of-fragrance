import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import time
import re
import os

def scraping_with_retry():
    retry_delay = 2  # seconds
    while True:
        try:
            base_url = 'https://www.fragrantica.com/'
            brand_csv = pd.read_csv('brand_urls.csv')

            for i in range(len(brand_csv['brand'])):
                brand = brand_csv['brand'][i]
                if not os.path.exists(brand):
                    os.mkdir(brand)
                print(f"Start scraping on {brand}")


                with open(f'brand_json/{brand}.json', 'r', encoding='utf-8') as file:
                    prod_brand_urls = json.load(file)


                scraping_no = 1
                for prod_url in prod_brand_urls:
                    url = base_url + prod_url

                    prod_url_name = prod_url.split("/")[-1].replace(".html", "")
                    if os.path.exists(f'{brand}/{brand}_{prod_url_name}_prod_dict.json'):
                        # print(f'{brand}/{brand}_{prod_url_name}_prod_dict.json')
                        scraping_no += 1
                        continue
                    else:
                        print(f"{scraping_no}/{len(prod_brand_urls)} of {brand}")

                    # code for scrapperapi scrap
                    # response = requests.get('http://api.scraperapi.com',
                    # {
                    #         'api_key': '03cb768180a77fe3746ee367fa787d4e',
                    #         'url': url,
                    #         'render': 'true'
                    #  }
                    # )

                    # code for abstract api to scrap
                    # response = requests.get('https://app.zenscrape.com/api/v1/get',
                    #                         headers={
                    #                             "apikey": "f9e1fad0-855b-11ee-909e-bbe99a7064b7"
                    #                         },
                    #                         params = (
                    #                            ("url",url),
                    #                            ("render", "true"),
                    #                            ("wait_for", "1"),
                    #                         )
                    #                         )

                    # code for scrapeops.io scrap

                    response = requests.get(
                        url='https://proxy.scrapeops.io/v1/',
                        params={
                            # 'api_key': "dcfbb9a9-56d1-4085-a3be-2291a907d570",
                            'api_key': "b33560d5-7b3c-4606-bdb3-ce3d82de97ed",
                            # 'api_key': "b58d9e17-a2e0-416e-a049-1f9493b4558f",
                            # 'api_key': "107ae881-59de-41e2-9640-c68c311a1c16",
                            # 'api_key': "add1bd93-1246-48a7-8706-d7668ba737ca",
                            # 'api_key': "cb8df83b-4863-4662-9625-453d0f7769fd",
                            # 'api_key': "d0537dde-9616-4f45-ac59-0d8aa976c3d7",
                            # 'api_key': "84f0227b-09a5-4fdf-8d97-123aa2ef3160",
                            'url': url,
                            'render_js': 'true',
                        },
                    )
                    soup = BeautifulSoup(response.text, 'html.parser')

                    list_of_perfume_dicts = []
                    # Create a new BeautifulSoup object from the HTML content
                    try:
                        perfume_name = soup.find_all("div", class_="cell small-12")[3].find_all("b")[0].get_text()
                        perfume_comp = soup.find_all("div", class_="cell small-12")[3].find_all("b")[1].get_text()
                        perfume_image = soup.find_all("div", class_="cell small-12")[1].find("img")["src"]
                        for_gender = soup.find("small").get_text()
                        print(perfume_name)
                    except:
                        print(soup.contents)
                        if "banned" in soup.contents[0] or "API" in soup.contents[0]:
                            exit()


                    try:
                        rating = float(soup.find("p", class_="info-note").find_all("span")[0].get_text())
                        number_votes = int(soup.find("p", class_="info-note").find_all("span")[2].get_text().replace(',', ''))
                    except:
                        rating = "NA"
                        number_votes = "NA"
                        print(f"{perfume_name} does not have a ranking")

                    try:
                        description = soup.find_all("div", class_="cell small-12")[3].find("p").get_text()
                        description = re.sub(r'\s+', ' ',description.strip())
                    except:
                        description = "NA"
                        print(f"{perfume_name} does not have a description")
                    ####### MAIN ACCORDS DICTIONARY #######

                    try:
                        main_accords = soup.find_all("div", class_="cell accord-box")
                        accords_dict = {}
                        for m in range(len(main_accords)):
                            accord_name = main_accords[m].get_text()
                            accord_value = float(main_accords[m].find("div", class_="accord-bar")["style"].rsplit("width: ")[1].strip("%;"))
                            accords_dict[accord_name] = accord_value
                    except:
                        accords_dict = {}
                        print(f"{perfume_name} does not have accords")

                    ####### FRAGRANCE NOTES #######
                    notes = soup.find_all("div", attrs={"style": "display: flex; flex-direction: column; justify-content: center; text-align: center; background: white;"})
                    top_notes_list = []
                    middle_notes_list = []
                    base_notes_list = []
                    try:
                        if notes[0].find_all("h4") == []:
                            note_div = notes[0].find('div', class_="text-center notes-box").find_next_sibling("div").find("div")
                            note_element = note_div.find('div')
                            for i in range(20):
                                try:
                                    middle_notes_list.append(note_element.get_text(strip=True))
                                    note_element = note_element.find_next_sibling("div")
                                except:
                                    break
                        else:
                            note_index = 0
                            for note in notes[0].find_all("h4"):
                                note_div = notes[0].find_all("h4")[note_index].find_next_sibling("div").find("div")
                                element_no = 0
                                for div in note_div:
                                    if note_index == 0:
                                        top_notes_list.append(div.get_text())
                                    elif note_index == 1:
                                        middle_notes_list.append(div.get_text())
                                    else:
                                        base_notes_list.append(div.get_text())
                                note_index += 1
                    except:
                        print(f"{perfume_name} does not have notes")
                    ####### VOTING DATA & INFORMATION #######
                    voting = soup.find_all("div", class_="cell small-1 medium-1 large-1")

                    ####### Longevity #######
                    long_v_weak = int(voting[0].get_text())
                    long_weak = int(voting[1].get_text())
                    long_moderate = int(voting[2].get_text())
                    long_long_last = int(voting[3].get_text())
                    long_eternal = int(voting[4].get_text())

                    ####### Sillage #######
                    sill_intimate = int(voting[5].get_text())
                    sill_moderate = int(voting[6].get_text())
                    sill_strong = int(voting[7].get_text())
                    sill_enormus = int(voting[8].get_text())

                    ####### Gender #######
                    gender_female = int(voting[9].get_text())
                    gender_more_fem = int(voting[10].get_text())
                    gender_unisex = int(voting[11].get_text())
                    gender_more_male = int(voting[12].get_text())
                    gender_male = int(voting[13].get_text())

                    ####### Price Value #######
                    value_w_over = int(voting[14].get_text())
                    value_over = int(voting[15].get_text())
                    value_ok = int(voting[16].get_text())
                    value_good = int(voting[17].get_text())
                    value_great = int(voting[18].get_text())
                    value_great = int(voting[18].get_text())

                    ####### CREATING THE DICTIONARY OF DATA #######
                    perfume_dict = {"name": perfume_name,
                                    "company": perfume_comp,
                                    "image": perfume_image,
                                    "for_gender": for_gender,
                                    "rating": rating,
                                    "number_votes": number_votes,
                                    "main accords": accords_dict,
                                    "description": description,
                                    "top notes": top_notes_list,
                                    "middle notes": middle_notes_list,
                                    "base notes": base_notes_list,
                                    "longevity":   {"very weak": long_v_weak,
                                                    "weak": long_weak,
                                                    "moderate": long_moderate,
                                                    "long lasting": long_long_last,
                                                    "eternal": long_eternal},
                                    "sillage":     {"intimate": sill_intimate,
                                                    "moderate": sill_moderate,
                                                    "strong": sill_strong,
                                                    "enormous": sill_enormus},
                                    "gender_vote": {"female": gender_female,
                                                    "more female": gender_more_fem,
                                                    "unisex": gender_unisex,
                                                    "more male": gender_more_male,
                                                    "male": gender_male},
                                    "price value": {"way overpriced": value_w_over,
                                                    "overpriced": value_over,
                                                    "ok": value_ok,
                                                    "good value": value_good,
                                                    "great value": value_great}
                                   }
                    scraping_no += 1
                    list_of_perfume_dicts.append(perfume_dict)
                    # time.sleep(2)
                    with open(f'{brand}/{brand}_{prod_url_name}_prod_dict.json', 'w') as file:
                        json.dump(perfume_dict, file, indent=4)
                        print(f"Finish scraping on {brand}_{prod_url_name}")
                print(f"Finish scraping on {brand}")
                print("---------------------------------")

        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

        else:
            # If no exception occurred, exit the loop and function
            return

#Call the function to run code with retry
scraping_with_retry()