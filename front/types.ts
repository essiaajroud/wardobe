
export enum Weather {
  SUMMER = 'summer',
  SPRING = 'spring',
  AUTUMN = 'autumn',
  WINTER = 'winter'
}

export enum Occasion {
  CASUAL = 'casual',
  WORK = 'work',
  PARTY = 'party',
  WEDDING = 'wedding',

}

export interface ClothingItem {
  id: string;
  imageData: string; // base64
  name: string;
  category: string;
  color: string;
  style: string;
  seasons: string[];
  description: string;
}

export interface OutfitRecommendation {
  items: string[]; // IDs of items
  reasoning: string;
  stylingTips: string;
}

export interface WeatherData {
  temperature: number;
  description: string;
  city: string;
  sources?: string[];
}
