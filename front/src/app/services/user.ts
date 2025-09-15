import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
@Injectable({
  providedIn: 'root'
})
export class UserService {
  private apiUrl = 'http://localhost:8081/users';

  constructor(private http: HttpClient) {}

  createUserFormData(formData: FormData): Observable<any> {
    return this.http.post(this.apiUrl, formData);
  }

  getAllUsers(): Observable<any[]> {
    return this.http.get<any[]>(this.apiUrl);
  }

updateUser(id: string, user: any, photo?: File): Observable<any> {
  const formData = new FormData();

  if (photo) {
    formData.append('file', photo);
  }

  const userCopy = { ...user };
  delete userCopy.photo;
  delete userCopy.photoUrl;

  formData.append('user', JSON.stringify(userCopy));

  return this.http.put(`${this.apiUrl}/${id}`, formData);
}




  deleteUser(id: string): Observable<void> {
    return this.http.delete<void>(`${this.apiUrl}/${id}`);
  }

  getUserById(id: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/${id}`);
  }
    getUserProfile(userId: string) {
    return this.http.get<any>(`${this.apiUrl}/profile/${userId}`);
  }
  getCurrentUserByEmail(email: string): Observable<any> {
  return this.http.get<any>(`${this.apiUrl}/me?email=${email}`);
}

forgotPassword(email: string): Observable<any> {
  return this.http.post(`${this.apiUrl}/forgot-password`, { email });
}

resetPassword(email: string, code: string, newPassword: string): Observable<any> {
  return this.http.post(`${this.apiUrl}/reset-password`, { email, code, newPassword });
}


}
