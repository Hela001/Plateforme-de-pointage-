import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Fraude {
  id: string;
  user_id: string;
  username: string;
  date: string;
  role: string;
  adresse: string;
  message: string;
  raisons?: string[];
}

@Injectable({
  providedIn: 'root'
})
export class FraudeService {
  private apiUrl = 'http://localhost:5010/fraudes';

  constructor(private http: HttpClient) {}

  getFraudes(): Observable<Fraude[]> {
    return this.http.get<Fraude[]>(this.apiUrl);
  }
  getFraudesFiltered(params: any): Observable<any> {
  return this.http.get<any>('http://localhost:5010/fraudes', { params });
}

}
