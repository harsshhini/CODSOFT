import random
class appt:
    datelist=[]
    apptDetails={}
    def dept(self):
        print("\nWhich area of SPECIFICATION are you looking for?\n1 Pediatric Primary care\n2 Pediatric Cardiology\n3 Pediatric Oncology \n4 Pediatric Gastroenterology\n5 Pediatric Neurology.\n6 Exit")
        self.choice1=int(input())
        return self.choice1
    def doctors_ppc(self,choice):
        print("\nDoctors: \n1 Mr. Vikram Sharma\n2 Ms. Anjali Patel\n3 EXIT")
        self.choice2=int(input())
        self.Dates(self.choice2,self.choice1,choice)
    def doctors_pc(self,choice):
        print("\nDoctors: \n1 Ms. Priya Gupta\n2 Mr. Arjun Verma\n3 EXIT")
        self.choice2=int(input())
        self.Dates(self.choice2,self.choice1,choice)
    def doctors_po(self,choice):
        print("\nDoctors: \n1 Mr. Siddharth Kapoor\n2 Ms. Aisha Mehta\n3 EXIT")
        self.choice2=int(input())
        self.Dates(self.choice2,self.choice1,choice)
    def doctors_pg(self,choice):
        print("\nDoctors: \n1 Ms. Pooja Malhotra\n2 Mr. Rohit Mishra\n3 EXIT")
        self.choice2=int(input())
        self.Dates(self.choice2,self.choice1,choice)
    def doctors_pn(self,choice):
        print("\nDoctors: \n1 Mr. Vivek Choudhary\n2 Ms. Neha Jain\n3 EXIT")
        self.choice2=int(input())
        self.Dates(self.choice2,self.choice1,choice)
    def Dates(self,choice2,choice1,choice):
        self.customerDetails=[]
        if choice==1:
            print("\nWhich DATE you are looking for?")
            self.date=int(input())
            self.datelist.append(self.date)
            print("\nName, Phone number:")
            self.namePhn=input()
            self.randomcode=random.randrange(1000,9999)
            self.customerDetails.extend([str(self.date)+"Jan",self.namePhn,(str(choice2)+str(choice1))])
            self.apptDetails[self.randomcode]=self.customerDetails
            print("\nYour details:\nDoctor Detail Code ",str(choice1)+str(choice2),"\nDate: ",self.date,"\nDetails: ",self.namePhn,"\nReferral code: ",self.randomcode)
            print(self.apptDetails)
        if choice==2:
            self.randomcode=int(input("referral code?"))
            if self.randomcode in self.apptDetails:
                print("Previous details will be deleted, Write your new Details: ")
                print("\nWhich DATE you are looking for?")
                self.date=int(input())
                self.datelist.append(self.date)
                print("\nName, Phone number:")
                self.namePhn=input()
                self.randomcode=random.randrange(1000,9999)
                self.customerDetails.extend([str(self.date)+"Jan",self.namePhn,(str(choice2)+str(choice1))])
                self.apptDetails[self.randomcode]=self.customerDetails
                print("\nYour details:\nDoctor Detail Code ",str(choice1)+str(choice2),"\nDate: ",self.date,"\nDetails: ",self.namePhn,"\nReferral code: ",self.randomcode)    
            else:
                print("Not Found")
        if choice==3:
            self.randomcode=int(input("referral code?"))
            if self.apptDetails(self.randomcode)!=None:

                self.apptDetails.pop(self.randomcode)
                print("Schedule deleted")
            else:
                print("NOT FOUND")


print("WELCOME TO CODSOFT PEDIATRIC MEDICAL SERVICES/nPlease enter the digit of your convinience\n1 Schedule Appointment\n2 Reschedule Appointment\n3 Cancel Appointment")
choice=int(input())
appointment=appt()
match(choice):
    case 1:
        choice1=appointment.dept()
        match(choice1):
            case(1):
                appointment.doctors_ppc(choice)
            case(2):
                appointment.doctors_pc(choice)
            case(3):
                appointment.doctors_po(choice)
            case(4):
                appointment.doctors_pg(choice)
            case(5):
                appointment.doctors_pn(choice)
    case 2:
        choice1=appointment.dept()
        match(choice1):
            case(1):
                appointment.doctors_ppc(choice)
            case(2):
                appointment.doctors_pc(choice)
            case(3):
                appointment.doctors_po(choice)
            case(4):
                appointment.doctors_pg(choice)
            case(5):
                appointment.doctors_pn(choice)
    case 3:
        choice1=appointment.dept()
        match(choice1):
            case(1):
                appointment.doctors_ppc(choice)
            case(2):
                appointment.doctors_pc(choice)
            case(3):
                appointment.doctors_po(choice)
            case(4):
                appointment.doctors_pg(choice)
            case(5):
                appointment.doctors_pn(choice)
           
