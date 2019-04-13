load semeion.data;
semeion(1,:);

x=semeion(:,1:256);
y=semeion(:,257:266);

row=1593;
correct=0;

for i=1:row


X=ones(1593,257);
X(:,2:257)=x;
Y=y;

testX=X(i,:);
testY=Y(i,:);

X(i,:)=[];
Y(i,:)=[];


theta=pinv(X'*X)*X'*Y;

h=testX*theta;


    m1=0;
    for j=1:10
        if h(1,j)>m1
            m1=h(1,j);
            n1=j;
        end
      end

      

    m2=0;
    for j=1:10
        if testY(1,j)>m2
            m2=testY(1,j);
            n2=j;
        end
    end
    
   
  

    if n1==n2
        correct=correct+1;
      end
      
end



accuracy=correct/row

    

