from llama_index.core.tools import FunctionTool
from utils.bigquery import BigQueryHandler
import json

bigquery_client = BigQueryHandler(
    project_id = 'tripadvisor-recommendations',
    credentials_path = "./sa.json"
)

def get_restaurant_recommendations(cuisine_filter:str=None, price_filter:str=None, top_k:int=3)->str:
    """
    This function return top k restaurants fit with user's need.
    
    Available params: (MUST be ONE of the value in the list)
        cuisine_filter: [Chinese, Indian, Italian, Japanese, Korean, Thai, Mexican, French, American, Mediterranean, Vietnamese, Spanish, Greek, Turkish, Brazilian, British, German, Caribbean, African, Middle Eastern, Barbecue, Sushi, Pizza, Seafood, Fast Food, Vegan, Dessert, Bakery, Grill, Cafe, Street Food]
        price_filter: [Cheap Eats, Mid-range, Fine Dining]
        
    Rules:
        - Input None to params if no information given
        - If user mention about number restaurant MUST input top_k, ortherwise skip
    """
    if top_k > 5:
        top_k = 5
    
    query = f"""
      with base_data as (
        select
          location_name,
          location_rank,
          location_overall_rate,
          review_count,
          price,
          location_url,
          location_map,
          string_agg(distinct cuisine_name, ', ') as cuisine_list
        from `tripadvisor-recommendations.dm_tripadvisor.fact_restaurant_feedback` as res
        left join `tripadvisor-recommendations.dm_tripadvisor.dim_cuisine` as cuisine
          on res.cuisine_id = cuisine.cuisine_id
        left join `tripadvisor-recommendations.dm_tripadvisor.dim_location` as loc
          on res.location_id = loc.location_id
        group by 1,2,3,4,5,6,7
      )

      select
        location_name,
        location_url,
        location_map
      from base_data
      where 1=1
      {f"and cuisine_list like '%{cuisine_filter}%'" if cuisine_filter else ""}
      {f"and price like '%{price_filter}%'" if price_filter else ""}
      order by location_rank, location_overall_rate, review_count desc
      limit {top_k}
    """
    return json.dumps(bigquery_client.fetch_bigquery(query).to_dict(orient="records"), indent=2)


get_restaurant_recommendations_tool = FunctionTool.from_defaults(fn=get_restaurant_recommendations)

if __name__ == '__main__':
    print(get_restaurant_recommendations(cuisine_filter="Barbecue"))