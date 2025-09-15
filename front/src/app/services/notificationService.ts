import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Notification {
  id?: string;
  userId: string;
  username: string;
  date: string;
  heure: string;
  statut: string;
  message: string;
  role: string;
  adresse: string;
}

@Injectable({
  providedIn: 'root'
})
export class NotificationService {

  private apiUrl = 'http://localhost:8081/notifications'; // Backend Spring Boot

  constructor(private http: HttpClient) {}

  getAllNotifications(): Observable<Notification[]> {
    return this.http.get<Notification[]>(this.apiUrl);
  }
  getNotificationById(id: string): Observable<Notification> {
  return this.http.get<Notification>(`${this.apiUrl}/${id}`);
}

}
