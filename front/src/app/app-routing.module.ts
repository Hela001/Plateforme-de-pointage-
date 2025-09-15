// Angular Import
import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';

// project import
import { AdminComponent } from './theme/layout/admin/admin.component';
import { GuestComponent } from './theme/layout/guest/guest.component';
import { roleGuard } from './guards/role-guard';

const routes: Routes = [
 {
  path: '',
  component: AdminComponent,
  children: [
    {
      path: '',
      redirectTo: '/auth/signin',
      pathMatch: 'full'
    },
    {
      path: 'analytics',
      loadComponent: () => import('./demo/dashboard/dash-analytics.component')
    },
    {
      path: 'component',
      loadChildren: () => import('./demo/ui-element/ui-basic.module').then((m) => m.UiBasicModule)
    },
    {
      path: 'chart',
      loadComponent: () => import('./demo/chart-maps/core-apex.component')
    },
    {
      path: 'forms/:id',
      loadComponent: () => import('./demo/forms/form-elements/form-elements.component').then(m => m.FormElementsComponent)
    },
    {
      path: 'users',
      loadComponent: () => import('./demo/tables/tbl-bootstrap/tbl-bootstrap.component').then(m => m.TblBootstrapComponent)
    },
   
    {
      path: 'pointage',
      loadComponent: () => import('./demo/pointage/pointage').then(m => m.PointageComponent)
    },
    
  {
  path: 'fraudes',
  loadComponent: () => import('./fraude-list/fraude-list').then(m => m.FraudeListComponent)
},
{
    path: 'chat',
    loadComponent: () => import('./chat-groq/chat-groq').then(m => m.ChatGroq)
  }

    ,
     {
      path: 'notifications',
      loadComponent: () => import('./demo/notifications/notifications').then(m => m.Notifications)
    },
    {
      path: 'profile',  
      loadComponent: () => import('./demo/profile/profile').then(m => m.Profile)
    }
  ]
}
,
  {
    path: '',
    component: GuestComponent,
    children: [
      {
           path: 'auth/signup',
           canActivate: [roleGuard],
           data: { roles: ['ADMIN', 'SITE_SUPERVISOR'] },
           loadComponent: () => import('./demo/pages/authentication/user/user').then(m => m.UserComponent)
      },

      {
        path: 'auth/signin',
        loadComponent: () => import('./demo/pages/authentication/sign-in/sign-in.component').then(m => m.SignInComponent)
      },
       {
      path: 'auth/forgot-password',
      loadComponent: () => import('./demo/forgot-password/forgot-password')
        .then(m => m.ForgotPasswordComponent)
    }
      ,      {
            path: 'profile',
          loadComponent: () => import('./demo/profile/profile').then(m => m.Profile)
     }
    ]
  }
  
 

];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule {}
