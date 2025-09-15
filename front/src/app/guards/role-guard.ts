import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { UserAuthService } from '../services/UserAuthService';


export const roleGuard: CanActivateFn = (route, state) => {
  const authService = inject(UserAuthService);
  const router = inject(Router);

  const allowedRoles = route.data?.['roles'] as string[];  
  const userRole = authService.getRole();

  if (allowedRoles && allowedRoles.includes(userRole)) {
    return true;
  }


  router.navigate(['/auth/signin']);
  return false;
};
