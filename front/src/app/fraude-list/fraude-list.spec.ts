import { ComponentFixture, TestBed } from '@angular/core/testing';

import { FraudeList } from './fraude-list';

describe('FraudeList', () => {
  let component: FraudeList;
  let fixture: ComponentFixture<FraudeList>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [FraudeList]
    })
    .compileComponents();

    fixture = TestBed.createComponent(FraudeList);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
