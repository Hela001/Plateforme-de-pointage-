// user-auth.service.ts
import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class UserAuthService {
     private apiUrl = 'http://localhost:8081/users';

      private roleKey = 'userRole';
  constructor(private http: HttpClient) {}
  setRole(role: string) {
    localStorage.setItem(this.roleKey, role);
  }

  getRole(): string {
    return localStorage.getItem(this.roleKey) || '';
  }
// setUser(user: any) {
//   localStorage.setItem('userId', user.id);
//   localStorage.setItem(this.roleKey, user.role);
// 
// }
 setUser(user: any) {
  localStorage.setItem('userId', user.id);
  localStorage.setItem('email', user.email);
  localStorage.setItem('username', user.username);
  localStorage.setItem(this.roleKey, user.role);
  localStorage.setItem('id', user.id); // utile pour /photo/{id}
}


getId(): string {
  return localStorage.getItem('userId') || '';
}


  clear() {
    localStorage.removeItem(this.roleKey);
  }
  signIn(email: string, password: string): Observable<any> {
  return this.http.post(`${this.apiUrl}/signin`, { email, password });
}

}
