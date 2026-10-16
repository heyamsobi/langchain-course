from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def main() -> None:
    print("Hello from langchain-course!")

    information = """
    Steven Paul Jobs (February 24, 1955 – October 5, 2011) was an American businessman, inventor, and investor. A pioneer of the personal computer revolution of the 1970s and 1980s, Jobs co-founded Apple Inc. with his early business partner Steve Wozniak as Apple Computer Company in 1976. After the company's board of directors fired him in 1985, he founded NeXT the same year and purchased Pixar in 1986, becoming its chairman and majority shareholder until 2007. Jobs returned to Apple in 1997 as CEO, where he was closely involved with the creation and promotion of many of the company's most influential products until his resignation in 2011.

Jobs was born in San Francisco in 1955 and adopted shortly afterward. He attended Reed College in 1972 before withdrawing that same year. In 1974, he traveled through India, seeking enlightenment before later studying Zen Buddhism. He and Wozniak co-founded Apple in 1976 to further develop and sell Wozniak's Apple I personal computer. Together, the duo gained fame and wealth a year later with the production and sale of the Apple II, one of the first highly successful mass-produced microcomputers.

Jobs saw the commercial potential of the Xerox Alto in 1979, which was mouse-driven and had a graphical user interface (GUI). This led to the development of the largely unsuccessful Apple Lisa in 1983, followed by the breakthrough Macintosh 128K in 1984, the first mass-produced computer with a GUI. The Macintosh launched the desktop publishing industry in 1985 (for example, the Aldus PageMaker) with the addition of the Apple LaserWriter, the first laser printer to feature vector graphics and PostScript.

In 1985, Jobs departed Apple after a long power struggle with the company's board and its then-CEO, John Sculley. That same year, Jobs took some Apple employees with him to found NeXT, a computer platform development company that specialized in computers for higher education and business markets, serving as its CEO. In 1986, he bought the computer graphics division of Lucasfilm, which was spun off independently as Pixar.[2] Pixar produced the first computer-animated feature film, Toy Story (1995), and became a leading animation studio, producing dozens of commercially successful and critically acclaimed films.

In 1997, Jobs returned to Apple as CEO after the company acquired NeXT. He was largely responsible for reviving Apple, which was on the verge of bankruptcy. He worked closely with British designer Jony Ive to develop a line of products and services that had larger cultural ramifications, beginning with the "Think different" advertising campaign and leading to the iMac, iTunes, Mac OS X, Apple Store, iPod, iTunes Store, iPhone, App Store, and iPad. Jobs was also a member of the board of directors at Gap Inc. from 1999 to 2002.[3] In 2003, Jobs was diagnosed with a pancreatic neuroendocrine tumor. He died of tumor-related respiratory arrest in 2011; in 2022, he was posthumously awarded the Presidential Medal of Freedom. Since his death, he has been granted 141 patents; Jobs holds over 450 patents in total.[4]

Early life and education
Family
Steven Paul Jobs[5] was born on February 24, 1955, in San Francisco, California, to Joanne Carole Schieble and Dr. Abdulfattah "John" Jandali (Arabic: عبد الفتاح الجندلي).[6] Abdulfattah Jandali was born in Syria into a wealthy Muslim household, the youngest of nine siblings. After obtaining his undergraduate degree at the American University of Beirut in Lebanon, Jandali pursued a PhD in political science at the University of Wisconsin. There, he met Joanne Schieble, an American Catholic of Swiss-German descent whose parents owned a mink farm and real estate in Green Bay. The two fell in love but faced opposition from Schieble's father due to Jandali's Muslim faith. When Schieble became pregnant, she arranged for a closed adoption, and travelled to San Francisco to give birth.[7]

Schieble requested that her son be adopted by college graduates. A lawyer and his wife were selected, but they withdrew after discovering that the baby was a boy, so Jobs was instead adopted by Paul Reinhold Jobs and his wife, Clara. Paul Jobs, an American of German descent, was the son of a dairy farmer from Washington County, Wisconsin. After dropping out of high school, Paul Jobs worked as a mechanic, then joined the US Coast Guard. After he was discharged in San Francisco, Paul married Clara Hagopian, of Armenian descent, in February 1946, and they moved to Wisconsin, then to Indiana, where Paul worked as a machinist and later as a car salesman. Because Clara missed San Francisco, she convinced Paul to move back. There, Paul worked as a repossession agent, and Clara became a bookkeeper. In 1955, after having an ectopic pregnancy, the couple looked to adopt a child.[7] Since they lacked a college education, Schieble initially refused to sign the adoption papers and went to court to request that her son be removed from the Jobs household and placed with a different family but changed her mind after Paul and Clara promised to pay for their son's college tuition.[7][8]
    """

    summary_template = """
    Given the information {information} about the person, I want you to create:
    1. A shprt summary.
    2. Two interesting facts about them.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash")
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
