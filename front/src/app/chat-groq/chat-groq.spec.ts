import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChatGroq } from './chat-groq';

describe('ChatGroq', () => {
  let component: ChatGroq;
  let fixture: ComponentFixture<ChatGroq>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ChatGroq]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ChatGroq);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
