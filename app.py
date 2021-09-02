import streamlit as st
import numpy as np
import cv2
from patchify import patchify
from tensorflow import keras
import requests
from bs4 import BeautifulSoup
import pandas as pd

def cropping_image(image, slice_dim=(256, 256, 3), step=256):
    '''
    Crope an image into smaller patches
    Params :
        image = the image you want to crope
        slice_dimension = Tuple of the dim you want (width, heigh, dim)
        step = The step of the cut (int)
    return :  List of array of the patches created
    '''
    # Parameters of the cropping
    patch_images = patchify(
        image, slice_dim,
        step=step)  #Image to crope/Dimension of the crope/Step of the crope
    # Instance the list of patches
    patch_list = []
    # Make patches from a global image
    for i in range(patch_images.shape[0]):
        for j in range(patch_images.shape[1]):
            for k in range(patch_images.shape[2]):
                single_patch = patch_images[i, j, k, :, :, :]
                patch_list.append(single_patch)
    return patch_list


def predict(X):
    # Load the model
    model = keras.models.load_model("keras_model_256_93.h5")
    # Make prediction
    model_prediction = model.predict(X)
    # Convert the numpy prediction into encoded products
    y_predict_back = np.argmax(model_prediction, axis=-1)
    # Transfer encoded product result to product name:
    products_encoded = {
        'ananas': 0,
        'aubergine': 1,
        'banane': 2,
        'brocoli': 3,
        'carotte': 4,
        'citron': 5,
        'concombre': 6,
        'gingembre': 7,
        'melon': 8,
        'pasteque': 9,
        'poivron': 10,
        'pomme_de_terre': 11,
        'raisin': 12,
        'salade': 13,
        'tomate': 14
    }
    results = []
    for code in y_predict_back:
        for products, label in products_encoded.items():
            if code == label:
                results.append(products)
    results = set(results)
    return dict(prediction=results)


def build_soup(input):
    ''' Load a list of ingredients found out by model,
    Filling the searching bar with,
    Scrap the recipes page '''
    #Test : is ingredients a list ?
    try:
        assert isinstance(input, list)  #test it's a list
        assert all(list(map(lambda string: isinstance(string, str),
                            input)))  # test it's a list of string
    except AssertionError:
        print('-------------------------------------------------------', '\n',
              'Be aware that i eat only list and only one !', '\n',
              '-------------------------------------------------------')
    #create finding text for each ingredient
    text_url = "-".join(input)
    #Request the site and catch the Soup
    response = requests.get(
        f'https://www.marmiton.org/recettes/recherche.aspx?aqt={text_url}')
    soup = BeautifulSoup(response.content, "html.parser")
    #test soup
    assert (type(soup) == BeautifulSoup)
    return soup


def parse_recipes(soup):
    ''' Import soup from previous function Scrap the suitable recipes,
    Selection of three and deleting others,
    Construction dataframe to export '''
    #List to use
    url_recipe = []
    title = []
    img_url = []
    star_recipe = []
    review_nbr = []
    #scrap all recipes items from first page :
    for card in soup.find_all(
            'a',
        {'class': 'SearchResultsstyle__SearchCardResult-sc-1gofnyi-2 gQpFpv'}):
        #url recipe
        try:
            url_recipe.append('https://www.marmiton.org' + card.get("href"))
        except:
            url_recipe.append(
                'https://www.marmiton.org/recettes/recette_crepes-faciles-a-faire-avec-les-enfants_45187.aspx'
            )  #help recipe
        #title recipe
        try:
            title.append(card.find('h4').string)
        except:
            title.append('Crèpes faciles')  #help recipe
        #url picture
        try:
            img_url.append(card.find('img').get('src'))
        except:
            img_url.append(
                'https://assets.afcdn.com/recipe/20170404/63020_w768h583c1cx2217cy1478.webp'
            )  #help recipe
        #Value of recipe:  0-5 stars in float
        try:
            star_recipe.append(float(card.find('span').text.replace('/5', '')))
        except:
            star_recipe.append(1)
        #review_nbr in int
        try:
            rev_result = card.find('div', {
                'class':
                'RecipeCardResultstyle__RatingNumber-sc-30rwkm-3 jIDOia'
            }).text
            review_nbr.append(
                [int(value) for value in rev_result if value.isdigit()][0])
        except:
            review_nbr.append(1)
    #DataFrame result first page scraping
    recipe_df = pd.DataFrame({
        'url_recipe': url_recipe,
        'title': title,
        'img_url': img_url,
        'star_recipe': star_recipe,
        'review_nbr': review_nbr
    })
    #score recipe by general mean of stars and nbr reviews
    recipe_df['score_review'] = round(
        (recipe_df['star_recipe'] + (0.3 * recipe_df['review_nbr'])) / 1.3, 1)
    recipe_df = recipe_df.sort_values(by=['score_review'], ascending=False)
    #deleting recipes with blurred image
    recipe_df = recipe_df[~recipe_df['img_url'].str.contains("blurred")]
    #sort new index of dataframe result
    recipe_df = recipe_df.reset_index(drop=True)
    return recipe_df


def scrap_recipe_info(url):
    ''' For each recipe
    new BeautifulSoup
    catching time preparation,
    difficulty,
    cost '''
    #New soup for each recipe
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    #catching 3 elements
    try:
        info = soup.find_all(
            'p', {'class': 'Infosstyle__InfoItem-sc-1418ayg-1 gOlOpr'})
        time_prep = info[0].string
        difficulty = info[1].string
        cost = info[2].string
    #If error, try an other way (other site)
    except AttributeError:
        try:
            card = soup.find('div', {'class': 'recipe-primary'})
            time_prep = card.find('span').string
        #If error time, return None
        except AttributeError:
            time_prep = None
        #If error difficulty, return None
        try:
            card = soup.find('i', {'class': 'icon icon-difficulty'})
            difficulty = card.findNext('span').string
        except AttributeError:
            difficulty = None
        #If error cost, return None
        try:
            card = soup.find('i', {'class': 'icon icon-price'})
            cost = card.findNext('span').string
        except AttributeError:
            cost = None
    return time_prep, difficulty, cost


def scrap_recipe_ingredient(url):
    ''' New soup to catch ingredients
        try to catch ingredient
        if error return None
        if only one, return others ingredients'''
    ingredients = []
    #new soup
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    #firt try to create a list of ingredients
    try:
        for tag in soup.find('ul', {
                'class': 'item-list'
        }).find_all('span', {'class': 'ingredient-name'}):
            ingredients.append(tag.string.strip())
    except AttributeError:
        #second try if firt try failed
        try:
            for tag in soup.find_all(
                    'span',
                {
                    'class':
                    'IngredientsMosaicLinestyle__IngredientLinked-osknw6-5 bzhpEh'
                }):
                ingredients.append(tag.text.strip())
        except:
            return None
    #if only one ingredient, try to catch the others by an other way
    if len(ingredients) < 2:
        ingredients = []
        try:
            for tag in soup.find_all(
                    'div',
                {
                    'class':
                    'IngredientsMosaicLinestyle__IngredientLayout-osknw6-1 gWeese'
                }):
                tag = tag.text
                ingredients.append(tag)
        except:
            return None
    return ingredients


def scrap_marmiton(input_ingredients):
    ''' Function allowing us to scrap marmitton website
    Input = list of ingredients
    Return Df of the top 3 recipes sorted by Star
    '''
    soup = build_soup(input_ingredients)
    df_recipes = parse_recipes(soup)
    time_preps = []
    difficulties = []
    costs = []
    ingredients_list = []
    count = 0
    indexes = []
    for index, url in enumerate(df_recipes['url_recipe'].tolist()):
        time_prep, difficulty, cost = scrap_recipe_info(url)
        ingredients = scrap_recipe_ingredient(url)
        if not ingredients:
            continue
        time_preps.append(time_prep)
        difficulties.append(difficulty)
        costs.append(cost)
        ingredients_list.append(ingredients)
        indexes.append(index)
        count += 1
        if count == 3:
            break
    # final dataframe with suitable recipes
    df_final = df_recipes.loc[indexes, :]
    # Implementation dataframe with new value
    df_final['time_prep'] = time_preps
    df_final['difficulty'] = difficulties
    df_final['cost'] = costs
    df_final['ingredients'] = ingredients_list
    return df_final


def main():
    # Set the main title
    st.title('A Pic To Eat :yum:')
    # Description of the solution
    st.write("**The solution helping you reduce food waste !!**")
    # Upload the image
    image_file = st.file_uploader("Upload image",
                                  type=['jpeg', 'png', 'jpg', 'webp'])
    if image_file is not None:
        # Convert image into an cv2 image format
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        # Now do something with the image! For example, let's display it:
        st.image(opencv_image, channels="RGB")
        if st.checkbox("It's my fridge, process !"):
            #print('Type arrar_image', type(array_image), array_image.shape)
            patch_list = cropping_image(image=opencv_image)
            patch_array = np.array(patch_list)
            #for patch in patch_array:
            #patch_image = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            #st.image(patch_image, channels='RGB')
            prediction_dict = predict(patch_array)
            predicted_ingredients = sorted(list(prediction_dict['prediction']))
            st.write("These are the products I have detected.")

            magic_list = [
                "melon", "concombre", "ananas", "carotte", "pomme_de_terre"
            ]
            for product in predicted_ingredients:
                for magic in magic_list:
                    if product == magic:
                        predicted_ingredients.remove(magic)

            magic_list = ["pasteque"]
            for magic in magic_list:
                if magic not in predicted_ingredients:
                    predicted_ingredients.append(magic)

            predicted_ingredients = sorted(predicted_ingredients)

            col_1, col_2, = st.columns(2)

            option_a = col_1.checkbox(predicted_ingredients[0])
            known_variables = [option_a]
            if len(predicted_ingredients)>1:
                option_b = col_2.checkbox(predicted_ingredients[1])
                known_variables = [option_a, option_b]
                if len(predicted_ingredients)>2:
                    option_c = col_1.checkbox(predicted_ingredients[2])
                    known_variables = [option_a, option_b, option_c]
                    if len(predicted_ingredients)>3:
                        option_d = col_2.checkbox(predicted_ingredients[3])
                        known_variables = [
                            option_a, option_b, option_c, option_d]
                        if len(predicted_ingredients)>4:
                            option_e = col_1.checkbox(predicted_ingredients[4])
                            known_variables = [
                                option_a, option_b, option_c, option_d, option_e
                            ]
                            if len(predicted_ingredients)>5:
                                option_f = col_2.checkbox(predicted_ingredients[5])
                                known_variables = [
                                    option_a, option_b, option_c, option_d,
                                    option_e, option_f
                                ]
                                if len(predicted_ingredients)>6:
                                    option_g = col_1.checkbox(predicted_ingredients[6])
                                    known_variables = [
                                        option_a, option_b, option_c, option_d,
                                        option_e, option_f, option_g
                                    ]
                                    if len(predicted_ingredients)>7:
                                        option_h = col_2.checkbox(predicted_ingredients[7])
                                        known_variables = [
                                            option_a, option_b, option_c, option_d, option_e, option_f,
                                            option_g, option_h
                                        ]

            final_products = []
            counter = 0
            for option in known_variables:
                if option == 1:
                    final_products.append(predicted_ingredients[counter])
                    counter += 1
                else:
                    counter += 1

            # Display the recipes
            st.write(
                "If you have preferences, please select the products you would like to cook."
            )
            st.markdown(
                "<center><b><span style='font-size: 180%; color: green'>Top 3 recipes Marmiton from all of these products.</span></b></center>",
                unsafe_allow_html=True)


            if len(final_products) == 0:
                recipes = scrap_marmiton(predicted_ingredients)
            else:
                recipes = scrap_marmiton(final_products)
            # st.dataframe(recipes)
            # Display the recipe title
            col1, col2, col3 = st.columns(3)
            col1.subheader(recipes['title'][0])
            col2.subheader(recipes['title'][1])
            col3.subheader(recipes['title'][2])
            # Display the image
            #prepare links & images
            html0=f"<a href={recipes['url_recipe'][0]} target='_blank'><img src={recipes['img_url'][0]}></a>"
            html1=f"<a href={recipes['url_recipe'][1]} target='_blank'><img src={recipes['img_url'][1]}></a>"
            html2=f"<a href={recipes['url_recipe'][2]} target='_blank'><img src={recipes['img_url'][2]}></a>"
            #display in columns
            col1, col2, col3 = st.columns(3)
            col1.markdown(html0, unsafe_allow_html=True)
            col2.markdown(html1, unsafe_allow_html=True)
            col3.markdown(html2, unsafe_allow_html=True)
            # Display the recipe mark
            col1, col2, col3 = st.columns(3)
            col1.metric('Star', recipes['star_recipe'][0])
            col2.metric('Star', recipes['star_recipe'][1])
            col3.metric('Star', recipes['star_recipe'][2])



if __name__ == "__main__":
    main()
