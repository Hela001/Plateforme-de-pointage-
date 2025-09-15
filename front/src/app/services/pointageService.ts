import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface PointageResponse {
  username?: string;
  statut?: string;
  adresse?: string;
  role?: string;
  fraude?: boolean;
  raisons?: string[];
}

export interface Pointage {
  _id: string;
  user_id: string;
  username: string;
  role: string;
  adresse: string;
  statut: string;
  date_pointage: string;
  heure_pointage?: string;
  image?: {
    type: string;
    data: string;
  };
  localisation?: {
    lat: number;
    lon: number;
  };
  arrivees?: { heure: Date;  }[];
  departs?: { heure: Date;}[];
}

@Injectable({ providedIn: 'root' })
export class PointageService {
  private apiUrl = 'http://localhost:5010/get_person_data';
  private backurl = 'http://localhost:8081/pointages'; 



  constructor(private http: HttpClient) {}

  fairePointage(lat: number, lon: number): Observable<PointageResponse> {
    const userId = localStorage.getItem('userId'); // peut être null

    const payload: any = { lat, lon };
    if (userId) {
      payload.userId = userId;
    }

    return this.http.post<PointageResponse>(this.apiUrl, payload);
  }

  
  getAllPointages(): Observable<Pointage[]> {
    return this.http.get<Pointage[]>(this.backurl);
  }
  getPointageById(id: string): Observable<Pointage> {
  return this.http.get<Pointage>(`${this.backurl}/${id}`);
}
detectFraude(file: File, username: string): Observable<any> {
  const formData = new FormData();
  formData.append('video', file);
  formData.append('username', username);

  return this.http.post<any>('http://localhost:5001/detect_fraude', formData);
}


}