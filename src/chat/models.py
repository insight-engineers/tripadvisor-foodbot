from typing import List

from pydantic import BaseModel


class RestaurantDescription(BaseModel):
    location_id: str
    short_description: str


class RestaurantsFinalized(BaseModel):
    begin_description: str
    restaurants: List[RestaurantDescription]
    end_description_with_follow_up: str
